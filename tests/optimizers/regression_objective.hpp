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
// parallel_reduce over the rows accumulating 1 + p sums, on a device backend one
// pass over X by chunks of rows and a small kernel that adds the chunks'
// partial sums, into buffers allocated at construction (functors with external
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

// The device route: one pass over X. A team takes a chunk of
// regression_chunk_rows rows: its threads and vector lanes compute the residuals,
// the value and the weights w_i (d s / d eta_i, kept in team scratch), then each
// thread takes columns and its vector lanes sum w_i X(i, j) over the chunk; the
// team writes its 1 + p partial sums to row `team` of a preallocated buffer. A
// second, small kernel adds the partials of the teams column by column, in a
// fixed order, so that the result does not depend on scheduling. An array
// reduction of 1 + p sums is not used: it asks CUDA for per-block scratch that
// grows with p, and at p = 201 it exceeded the limit ("requested too much L0
// scratch memory"). The buffers are allocated once, at construction.
inline constexpr int regression_chunk_rows = 256;

template <class ExecSpace>
struct regression_chunk_functor {
  using memory_space = typename ExecSpace::memory_space;
  using member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using scratch_view = Kokkos::View<double*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  regression_sums_functor<memory_space> f;
  Kokkos::View<double**, Kokkos::LayoutLeft, memory_space> partial;  // teams x (1 + p)
  KOKKOS_INLINE_FUNCTION void operator()(const member& team) const {
    const int n = static_cast<int>(f.y.extent(0));
    const int i0 = team.league_rank() * regression_chunk_rows;
    const int rows = n - i0 < regression_chunk_rows ? n - i0 : regression_chunk_rows;
    scratch_view w(team.team_scratch(0), regression_chunk_rows);
    double value = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(team, rows),
        [&](int k, double& s) {
          const int i = i0 + k;
          double eta = 0.0;
          for (int j = 0; j < f.p; ++j) eta += f.X(i, j) * f.beta(j);
          const double r = f.y(i) - eta;
          if (f.loss == regression_loss::gaussian) {
            s += 0.5 * r * r * f.inv_sigma * f.inv_sigma;
            w(k) = -r * f.inv_sigma * f.inv_sigma;
          } else {
            s += Kokkos::fabs(r) * f.inv_sigma;
            w(k) = r > 0.0 ? -f.inv_sigma : (r < 0.0 ? f.inv_sigma : 0.0);
          }
        },
        value);
    Kokkos::single(Kokkos::PerTeam(team), [&]() { partial(team.league_rank(), 0) = value; });
    if (f.value_count == 1) return;
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, f.p), [&](int j) {
      double s = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(team, rows), [&](int k, double& a) { a += w(k) * f.X(i0 + k, j); }, s);
      Kokkos::single(Kokkos::PerThread(team), [&]() { partial(team.league_rank(), 1 + j) = s; });
    });
  }
};

template <class MemSpace>
struct regression_combine_functor {
  Kokkos::View<const double**, Kokkos::LayoutLeft, MemSpace> partial;
  Kokkos::View<double*, MemSpace> out;
  KOKKOS_INLINE_FUNCTION void operator()(int k) const {
    double s = 0.0;
    for (int t = 0; t < static_cast<int>(partial.extent(0)); ++t) s += partial(t, k);
    out(k) = s;
  }
};

template <class MemSpace>
struct regression_device_buffers {
  Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace> partial;
  Kokkos::View<double*, MemSpace> out;
  typename Kokkos::View<double*, MemSpace>::host_mirror_type out_host;
};

template <class ExecSpace>
void regression_sums(const regression_sums_functor<typename ExecSpace::memory_space>& f,
                     const regression_device_buffers<typename ExecSpace::memory_space>& buffers, double* out) {
  using memory_space = typename ExecSpace::memory_space;
  const int n = static_cast<int>(f.y.extent(0));
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible) {
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> r(out, f.value_count);
    Kokkos::parallel_reduce("regression_test_sums", Kokkos::RangePolicy<ExecSpace>(0, n), f, r);
    Kokkos::fence();
  } else {
    using functor = regression_chunk_functor<ExecSpace>;
    const int teams = static_cast<int>(buffers.partial.extent(0));
    const auto bytes = functor::scratch_view::shmem_size(regression_chunk_rows);
    Kokkos::parallel_for("regression_test_chunks",
                         Kokkos::TeamPolicy<ExecSpace>(teams, Kokkos::AUTO, 32).set_scratch_size(0, Kokkos::PerTeam(bytes)),
                         functor{f, buffers.partial});
    Kokkos::parallel_for("regression_test_combine", Kokkos::RangePolicy<ExecSpace>(0, f.value_count),
                         regression_combine_functor<memory_space>{buffers.partial, buffers.out});
    const auto n_out = std::make_pair(0, f.value_count);
    Kokkos::deep_copy(Kokkos::subview(buffers.out_host, n_out), Kokkos::subview(buffers.out, n_out));
    for (int k = 0; k < f.value_count; ++k) out[k] = buffers.out_host(k);
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
    if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible) {
      buffers_.partial = decltype(buffers_.partial)("regression_test_partial",
                                                    (n + regression_chunk_rows - 1) / regression_chunk_rows, p + 1);
      buffers_.out = decltype(buffers_.out)("regression_test_out", p + 1);
      buffers_.out_host = Kokkos::create_mirror_view(buffers_.out);
    }
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
    regression_sums<ExecSpace>(f, buffers_, sums_.data());
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
  regression_device_buffers<memory_space> buffers_;  // device route only
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
