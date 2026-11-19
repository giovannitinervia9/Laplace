// The fit_distrib bench (note/optimizers.md, section 8): the negative
// log-likelihood of iid data y under a distrib D, as an objective of the
// optimizers in free coordinates u, theta_j = h_j(u_j) with a default scalar
// bijector per parameter domain. A test bench, not a public API: the engine of
// phase 7 replaces it.
//
//   f(u) = -sum_i l(theta(u); y_i)                       (summed, not averaged)
//   g_j  = -S_j h_j'(u_j),        S_j  = sum_i dl/dtheta_j
//   H_jk = -S_jk h_j' h_k' - delta_jk S_j h_j''     observed (S_jk = sum_i d2l/dtheta_j dtheta_k)
//   H_jk = n I_jk(theta) h_j' h_k'                  expected (Fisher scoring)
//
// Every member is one parallel_reduce over the observations, accumulating
// 1 (value), 1 + P (gradient) or 1 + P + P(P+1)/2 sums (observed curvature);
// the expected curvature adds no per-observation work: I(theta) is evaluated
// once on the host (distrib_expected_info). A parameter with regularity r_j < 2
// (mu of laplace1, laplace2) has no usable observed second derivative, so the
// observed curvature is refused at construction for such a distrib.
//
// D's non-parameter fields (binomial's size) come from a builder that writes
// theta into a D: the default one aggregate-initializes D{theta_0, ...}, which
// fits every distrib whose parameters are its only fields; binomial gets
// binomial_builder, which carries its size.
#pragma once

#include <laplace/bijector/scalar.hpp>
#include <laplace/dense/dense.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/binomial.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/optimizers/newton.hpp>
#include <laplace/optimizers/objective.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fit_distrib_bench {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
using laplace::bijector::scalar_bijector;
using laplace::bijector::scalar_bijector_map;

// identity on R, log on (0, inf), logit on (0, 1), the shifted/bounded maps otherwise
inline scalar_bijector default_bijector(const ld::distrib_param_info& p) {
  const bool lo_fin = std::isfinite(p.lower), hi_fin = std::isfinite(p.upper);
  if (!lo_fin && !hi_fin) return scalar_bijector{scalar_bijector_map::identity};
  if (lo_fin && !hi_fin)
    return p.lower == 0.0 ? scalar_bijector{scalar_bijector_map::log} : scalar_bijector::lower_bounded(p.lower);
  if (!lo_fin) return scalar_bijector::upper_bounded(p.upper);
  if (p.lower == 0.0 && p.upper == 1.0) return scalar_bijector{scalar_bijector_map::logit};
  return scalar_bijector::bounded(p.lower, p.upper);
}

// D{theta_0, ..., theta_(P-1)}
template <class D>
struct aggregate_builder {
  template <std::size_t... I>
  LAPLACE_HD static D make(const double* t, std::index_sequence<I...>) {
    return D{t[I]...};
  }
  LAPLACE_HD D operator()(const double* theta) const { return make(theta, std::make_index_sequence<D::n_params>{}); }
};

// binomial: the size is data, carried by the builder
struct binomial_builder {
  double size;
  LAPLACE_HD ld::binomial_distrib operator()(const double* theta) const { return {theta[0], size}; }
};

// what one pass accumulates: 0 value, 1 also the score, 2 also the observed second derivatives
template <class D, class Builder, class MemSpace, int Order>
struct loglik_sums_functor {
  static constexpr int P = D::n_params;
  static constexpr int count = 1 + (Order >= 1 ? P : 0) + (Order >= 2 ? P * (P + 1) / 2 : 0);
  using value_type = double[];
  using size_type = int;
  const size_type value_count = count;

  Kokkos::View<const double*, MemSpace> y;
  Builder build;
  double theta[P];

  KOKKOS_INLINE_FUNCTION void operator()(int i, value_type s) const {
    const D d = build(theta);
    s[0] += ld::detail_generic::log_density(d, y(i));
    if constexpr (Order >= 1) {
      const auto d1 = ld::detail_generic::log_density_dk<1>(d, y(i));
      for (int j = 0; j < P; ++j) s[1 + j] += d1[j];
    }
    if constexpr (Order >= 2) {
      const auto d2 = ld::detail_generic::log_density_dk<2>(d, y(i));
      for (int k = 0; k < P * (P + 1) / 2; ++k) s[1 + P + k] += d2[k];
    }
  }
  KOKKOS_INLINE_FUNCTION void init(value_type s) const {
    for (int k = 0; k < count; ++k) s[k] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (int k = 0; k < count; ++k) dst[k] += src[k];
  }
};

// one reduction (a named function with external linkage: CLAUDE.md trap)
template <int Order, class D, class Builder, class ExecSpace>
void loglik_sums(const Kokkos::View<const double*, typename ExecSpace::memory_space>& y, const Builder& build,
                 const double* theta, double* out) {
  using F = loglik_sums_functor<D, Builder, typename ExecSpace::memory_space, Order>;
  F f{.y = y, .build = build};
  for (int j = 0; j < D::n_params; ++j) f.theta[j] = theta[j];
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> r(out, F::count);
  Kokkos::parallel_reduce("fit_distrib_loglik", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(y.extent(0))), f, r);
  Kokkos::fence();
}

enum class fit_curvature { observed, expected };

// Fisher scoring as the bench runs it: Newton on the expected curvature with every step
// capped at 1 in the free coordinates (Giovanni, 2026-11-19; newton_optimizer's own default
// stays infinite). Measured over the 36 data sets of the bench at n = 1e3 and 1e5, from the
// library start and the +1 start (144 runs, the diagnostic "Fisher scoring max_step" of
// test_fit_distrib_diagnostics.cpp): converged 131 / 133 / 134 with max_step infinite / 2 / 1, iterations
// 1442 / 1446 / 1153, value evaluations 6089 / 6414 / 4102, no run changed to a worse status.
// Without the cap the full step on a small expected curvature in a shape's free coordinate
// sends pseudohuber's nu to 1.5e-4 and then 1e28 from the +1 start.
constexpr double fisher_max_step = 1.0;
inline laplace::optimizers::newton_optimizer<> fisher_scoring_optimizer() {
  laplace::optimizers::newton_optimizer<> newton;
  newton.max_step = fisher_max_step;
  return newton;
}

template <class D, class Builder = aggregate_builder<D>, class ExecSpace = Kokkos::DefaultExecutionSpace>
struct fit_distrib_objective {
  static constexpr int P = D::n_params;
  using distrib_type = D;
  using curvature_type = lo::dense_curvature_matrix;
  using data_view = Kokkos::View<const double*, typename ExecSpace::memory_space>;

  data_view y;
  Builder build{};
  fit_curvature curvature = fit_curvature::observed;
  scalar_bijector h[P];
  long passes = 0;  // reductions launched (one per member call)

  fit_distrib_objective(data_view y_, fit_curvature c, Builder b = {}) : y(y_), build(b), curvature(c) {
    for (int j = 0; j < P; ++j) h[j] = default_bijector(D::params[j]);
    if (c == fit_curvature::observed)
      for (int j = 0; j < P; ++j)
        if (D::params[j].regularity < 2)
          throw std::invalid_argument("fit_distrib_objective: observed curvature refused for a parameter with r_j < 2");
  }

  // theta = h(u); false when some theta_j falls outside its domain (a refused point)
  bool theta_of(const lo::optimizer_point& u, double* theta) const {
    for (int j = 0; j < P; ++j) {
      theta[j] = h[j].inverse(u(j));
      if (!(theta[j] > D::params[j].lower && theta[j] < D::params[j].upper)) return false;
    }
    return true;
  }
  D distrib_at(const lo::optimizer_point& u) const {
    double t[P];
    theta_of(u, t);
    return build(t);
  }
  // The estimate lies on the boundary of the domain (Giovanni, 2026-11-15): some parameter
  // with a finite end of its domain has a standard error above 10 in its free coordinate,
  // sqrt((H^-1)_jj) with H the expected information in u. The free coordinates of such
  // parameters are log or logit maps, so the measure is relative and scale-free. At an MLE
  // on the boundary (a sample of zeros under a bernoulli) l flattens as u goes to infinity
  // and the standard error grows without bound (measured 860 and above), while genuine
  // fits stay at about sqrt(2/n) whatever the scale of theta (at most 1.41, at n = 2).
  // Parameters on all of R have no end and are not checked. The optimizer's verdict is
  // about f; this flag is about the fit.
  bool on_boundary(const lo::optimizer_point& u) const {
    const auto se = free_standard_errors(u);
    for (int j = 0; j < P; ++j)
      if ((std::isfinite(D::params[j].lower) || std::isfinite(D::params[j].upper)) && !(se[j] <= 10.0)) return true;
    return false;
  }

  // sqrt((H^-1)_jj) with H the expected curvature n I(theta) h' h'^T in u, for
  // any number of parameters: H is symmetric positive definite, so the diagonal
  // of its inverse comes from one Cholesky and P triangular solves on the
  // identity (laplace::dense). The first version inverted a 2 x 2 by hand in a
  // double H[2][2] indexed by P, which wrote out of bounds for a family with
  // three parameters (found on 2026-11-15 by the skew normal).
  std::vector<double> free_standard_errors(const lo::optimizer_point& u) const {
    double t[P];
    theta_of(u, t);
    const auto I = ld::distrib_expected_info<0>(build(t));
    const double n = static_cast<double>(y.extent(0));
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> H("H", P, P), L("L", P, P), X("X", P, P);
    for (int j = 0; j < P; ++j)
      for (int k = 0; k < P; ++k) {
        const int idx[2] = {j < k ? j : k, j < k ? k : j};
        H(j, k) = n * I[ld::sym_tensor_position<P, 2>(idx)] * h[j].inverse_d1(u(j)) * h[k].inverse_d1(u(k));
        X(j, k) = j == k ? 1.0 : 0.0;
      }
    std::vector<double> se(P, std::numeric_limits<double>::infinity());
    if (!laplace::dense::spd_solve(H, L, X).success) return se;  // not positive definite: no finite error
    for (int j = 0; j < P; ++j) se[j] = std::sqrt(X(j, j));
    return se;
  }

  // u = g(theta)
  void free_of(const double* theta, const lo::optimizer_vector& u) const {
    for (int j = 0; j < P; ++j) u(j) = h[j].forward(theta[j]);
  }

  double value(const lo::optimizer_point& u) {
    double t[P], s[1];
    if (!theta_of(u, t)) return std::numeric_limits<double>::quiet_NaN();
    loglik_sums<0, D, Builder, ExecSpace>(y, build, t, s);
    ++passes;
    return -s[0];
  }

  double value_gradient(const lo::optimizer_point& u, const lo::optimizer_vector& g) {
    double t[P], s[1 + P];
    if (!theta_of(u, t)) return std::numeric_limits<double>::quiet_NaN();
    loglik_sums<1, D, Builder, ExecSpace>(y, build, t, s);
    ++passes;
    for (int j = 0; j < P; ++j) g(j) = -s[1 + j] * h[j].inverse_d1(u(j));
    return -s[0];
  }

  double value_gradient_curvature(const lo::optimizer_point& u, const lo::optimizer_vector& g, curvature_type& H) {
    double t[P], d1[P];
    if (!theta_of(u, t)) return std::numeric_limits<double>::quiet_NaN();
    for (int j = 0; j < P; ++j) d1[j] = h[j].inverse_d1(u(j));
    if (curvature == fit_curvature::expected) {
      const double f = value_gradient(u, g);
      const auto I = ld::distrib_expected_info<0>(build(t));
      const double n = static_cast<double>(y.extent(0));
      for (int j = 0; j < P; ++j)
        for (int k = j; k < P; ++k) {
          const int idx[2] = {j, k};
          H(j, k) = H(k, j) = n * I[ld::sym_tensor_position<P, 2>(idx)] * d1[j] * d1[k];
        }
      return f;
    }
    double s[1 + P + P * (P + 1) / 2];
    loglik_sums<2, D, Builder, ExecSpace>(y, build, t, s);
    ++passes;
    for (int j = 0; j < P; ++j) g(j) = -s[1 + j] * d1[j];
    for (int j = 0; j < P; ++j)
      for (int k = j; k < P; ++k) {
        const int idx[2] = {j, k};
        double v = -s[1 + P + ld::sym_tensor_position<P, 2>(idx)] * d1[j] * d1[k];
        if (j == k) v -= s[1 + j] * h[j].inverse_d2(u(j));
        H(j, k) = H(k, j) = v;
      }
    return -s[0];
  }
};

}  // namespace fit_distrib_bench
