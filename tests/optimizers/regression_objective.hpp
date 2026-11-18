#pragma once

// A regression test objective for the proximal methods (note/optimizers.md,
// section 12, decision 8), in the tests only; shared with the bundle tests.
//
//   gaussian: s(beta) = sum_i (y_i - x_i'beta)^2 / (2 sigma^2),  H = X'X / sigma^2
//   laplace:  s(beta) = sum_i |y_i - x_i'beta| / sigma,  a subgradient
//             -sum_i sign(r_i) x_i / sigma (sign(0) = 0), and as curvature the
//             expected information X'X / sigma^2 (Fisher scoring; not the observed
//             curvature, which is 0 off the kinks)
//
// -l up to a constant in beta, summed and not divided by n (the objective's
// units of decision 5). X is dense, n x p, column-major in the execution
// space's memory; on a host backend each value and gradient is one
// parallel_reduce over the rows accumulating 1 + p sums, on a device backend a
// scalar reduction and a team reduction per column (functors with external
// linkage, CLAUDE.md trap); X'X is formed once at construction.

#include <laplace/optimizers/objective.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace regression_test {

enum class regression_loss { gaussian, laplace };

template <class MemSpace>
struct regression_sums_functor {
  using value_type = double[];
  using size_type = int;
  size_type value_count;  // 1 + p with the gradient, 1 without
  Kokkos::View<const double**, Kokkos::LayoutLeft, MemSpace> X;
  Kokkos::View<const double*, MemSpace> y, beta;
  regression_loss loss;
  double inv_sigma;
  int p;

  KOKKOS_INLINE_FUNCTION void operator()(int i, value_type s) const {
    double eta = 0.0;
    for (int j = 0; j < p; ++j) eta += X(i, j) * beta(j);
    const double r = y(i) - eta;
    double w;
    if (loss == regression_loss::gaussian) {
      s[0] += 0.5 * r * r * inv_sigma * inv_sigma;
      w = -r * inv_sigma * inv_sigma;
    } else {
      s[0] += Kokkos::fabs(r) * inv_sigma;
      w = r > 0.0 ? -inv_sigma : (r < 0.0 ? inv_sigma : 0.0);
    }
    if (value_count > 1)
      for (int j = 0; j < p; ++j) s[1 + j] += w * X(i, j);
  }
  KOKKOS_INLINE_FUNCTION void init(value_type s) const {
    for (int k = 0; k < value_count; ++k) s[k] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (int k = 0; k < value_count; ++k) dst[k] += src[k];
  }
};

// The device route: the value and the weights w_i (d s / d eta_i) in one scalar
// reduction, then each gradient entry sum_i w_i X(i, j) by a team per column. An
// array reduction of 1 + p sums asks CUDA for per-block scratch that grows with p,
// and at p = 201 it exceeded the limit ("requested too much L0 scratch memory").
template <class MemSpace>
struct regression_value_weights_functor {
  regression_sums_functor<MemSpace> f;
  Kokkos::View<double*, MemSpace> w;
  KOKKOS_INLINE_FUNCTION void operator()(int i, double& s) const {
    double eta = 0.0;
    for (int j = 0; j < f.p; ++j) eta += f.X(i, j) * f.beta(j);
    const double r = f.y(i) - eta;
    if (f.loss == regression_loss::gaussian) {
      s += 0.5 * r * r * f.inv_sigma * f.inv_sigma;
      w(i) = -r * f.inv_sigma * f.inv_sigma;
    } else {
      s += Kokkos::fabs(r) * f.inv_sigma;
      w(i) = r > 0.0 ? -f.inv_sigma : (r < 0.0 ? f.inv_sigma : 0.0);
    }
  }
};

template <class ExecSpace>
struct regression_gradient_functor {
  using memory_space = typename ExecSpace::memory_space;
  using member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  Kokkos::View<const double**, Kokkos::LayoutLeft, memory_space> X;
  Kokkos::View<const double*, memory_space> w;
  Kokkos::View<double*, memory_space> g;
  KOKKOS_INLINE_FUNCTION void operator()(const member& team) const {
    const int j = team.league_rank();
    double s = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, static_cast<int>(w.extent(0))), [&](int i, double& a) { a += w(i) * X(i, j); }, s);
    Kokkos::single(Kokkos::PerTeam(team), [&]() { g(j) = s; });
  }
};

template <class ExecSpace>
void regression_sums(const regression_sums_functor<typename ExecSpace::memory_space>& f, double* out) {
  using memory_space = typename ExecSpace::memory_space;
  const int n = static_cast<int>(f.y.extent(0));
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible) {
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> r(out, f.value_count);
    Kokkos::parallel_reduce("regression_test_sums", Kokkos::RangePolicy<ExecSpace>(0, n), f, r);
    Kokkos::fence();
  } else {
    Kokkos::View<double*, memory_space> w("regression_test_w", n), g("regression_test_g", f.p);
    double value = 0.0;
    Kokkos::parallel_reduce("regression_test_value", Kokkos::RangePolicy<ExecSpace>(0, n),
                            regression_value_weights_functor<memory_space>{f, w}, value);
    out[0] = value;
    if (f.value_count > 1) {
      Kokkos::parallel_for("regression_test_gradient", Kokkos::TeamPolicy<ExecSpace>(f.p, Kokkos::AUTO),
                           regression_gradient_functor<ExecSpace>{f.X, w, g});
      auto gh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g);
      for (int j = 0; j < f.p; ++j) out[1 + j] = gh(j);
    }
    Kokkos::fence();
  }
}

// a small deterministic generator (splitmix64), the same on every platform
struct regression_rng {
  std::uint64_t state;
  double uniform() {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (static_cast<double>(z >> 11) + 0.5) * 0x1.0p-53;
  }
  double normal() {  // Box-Muller
    const double u1 = uniform(), u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

template <class ExecSpace = Kokkos::DefaultExecutionSpace>
class regression_test_objective {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using curvature_type = laplace::optimizers::dense_curvature_matrix;

  // X n x p column-major on the host (x_host[i + n j]), y of length n
  regression_test_objective(const std::vector<double>& x_host, const std::vector<double>& y_host, int n, int p,
                            regression_loss loss, double sigma = 1.0)
      : n_(n), p_(p), loss_(loss), inv_sigma_(1.0 / sigma), X_("X", n, p), y_("y", n), beta_("beta", p),
        beta_host_("beta_host", p), XtX_("XtX", p, p), sums_(static_cast<std::size_t>(p) + 1) {
    auto Xh = Kokkos::create_mirror_view(X_);
    auto yh = Kokkos::create_mirror_view(y_);
    for (int j = 0; j < p; ++j)
      for (int i = 0; i < n; ++i) Xh(i, j) = x_host[std::size_t(i) + std::size_t(n) * std::size_t(j)];
    for (int i = 0; i < n; ++i) yh(i) = y_host[std::size_t(i)];
    Kokkos::deep_copy(X_, Xh);
    Kokkos::deep_copy(y_, yh);
    for (int a = 0; a < p; ++a)
      for (int b = 0; b <= a; ++b) {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
          s += x_host[std::size_t(i) + std::size_t(n) * std::size_t(a)] * x_host[std::size_t(i) + std::size_t(n) * std::size_t(b)];
        XtX_(a, b) = XtX_(b, a) = s * inv_sigma_ * inv_sigma_;
      }
  }

  double value(const laplace::optimizers::optimizer_point& beta) { return evaluate(beta, false); }

  double value_gradient(const laplace::optimizers::optimizer_point& beta, const laplace::optimizers::optimizer_vector& g) {
    const double f = evaluate(beta, true);
    for (int j = 0; j < p_; ++j) g(j) = sums_[std::size_t(j) + 1];
    return f;
  }

  double value_gradient_curvature(const laplace::optimizers::optimizer_point& beta,
                                  const laplace::optimizers::optimizer_vector& g, curvature_type& H) {
    const double f = value_gradient(beta, g);
    Kokkos::deep_copy(H, XtX_);
    return f;
  }

  int rows() const { return n_; }
  int columns() const { return p_; }

 private:
  double evaluate(const laplace::optimizers::optimizer_point& beta, bool gradient) {
    for (int j = 0; j < p_; ++j) beta_host_(j) = beta(j);
    Kokkos::deep_copy(beta_, beta_host_);
    regression_sums_functor<memory_space> f{gradient ? p_ + 1 : 1, X_, y_, beta_, loss_, inv_sigma_, p_};
    regression_sums<ExecSpace>(f, sums_.data());
    return sums_[0];
  }

  int n_, p_;
  regression_loss loss_;
  double inv_sigma_;
  Kokkos::View<double**, Kokkos::LayoutLeft, memory_space> X_;
  Kokkos::View<double*, memory_space> y_, beta_;
  Kokkos::View<double*, Kokkos::HostSpace> beta_host_;
  curvature_type XtX_;
  std::vector<double> sums_;
};

// A test problem: X with a column of ones first, then p - 1 standard normal
// columns, equicorrelated with correlation `correlation` (a shared factor);
// y = X beta_true + noise, beta_true with `active` nonzero slopes.
struct regression_problem {
  int n, p;
  std::vector<double> x, y, beta_true;
};

inline regression_problem make_regression_problem(int n, int p, int active, double noise, std::uint64_t seed,
                                                  bool heavy_tails = false, double correlation = 0.0) {
  regression_problem pr{n, p, std::vector<double>(std::size_t(n) * std::size_t(p)), std::vector<double>(std::size_t(n)),
                        std::vector<double>(std::size_t(p), 0.0)};
  regression_rng rng{seed};
  for (int i = 0; i < n; ++i) pr.x[std::size_t(i)] = 1.0;
  std::vector<double> factor(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) factor[std::size_t(i)] = rng.normal();
  const double a = std::sqrt(correlation), b = std::sqrt(1.0 - correlation);
  for (int j = 1; j < p; ++j)
    for (int i = 0; i < n; ++i)
      pr.x[std::size_t(i) + std::size_t(n) * std::size_t(j)] = a * factor[std::size_t(i)] + b * rng.normal();
  pr.beta_true[0] = 0.5;
  for (int j = 1; j <= active && j < p; ++j) pr.beta_true[std::size_t(j)] = (j % 2 ? 1.0 : -1.0) * (0.5 + 0.25 * j);
  for (int i = 0; i < n; ++i) {
    double eta = 0.0;
    for (int j = 0; j < p; ++j) eta += pr.x[std::size_t(i) + std::size_t(n) * std::size_t(j)] * pr.beta_true[std::size_t(j)];
    const double e = heavy_tails ? std::log(rng.uniform() / rng.uniform()) : rng.normal();  // Laplace(0, 1) or N(0, 1)
    pr.y[std::size_t(i)] = eta + noise * e;
  }
  return pr;
}

}  // namespace regression_test
