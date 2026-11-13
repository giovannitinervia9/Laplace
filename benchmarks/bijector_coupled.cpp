// Speed of the coupled bijectors (include/laplace/bijector/: alr,
// log_cholesky, partial_corr, matrix_log): ns per call of the map, of its derivative of
// orders 1, 3, 6 along the tuple (0, ..., 0), and of the log-Jacobian and its
// order-6 derivative, in both directions, on one host thread (Serial) and on
// the default execution space (OpenMP or CUDA). One call writes every output
// of that tuple. The inverse runs at u_k = x + 0.1 k, the forward at a fixed
// interior point moved by x/1000, x uniform in [-3, 3]. Each call has its own
// row of a scratch View in the space's memory, as a caller on the device
// would provide.
#include "measure.hpp"

#include <laplace/bijector/alr.hpp>
#include <laplace/bijector/log_cholesky.hpp>
#include <laplace/bijector/matrix_log.hpp>
#include <laplace/bijector/partial_corr.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <type_traits>
#include <vector>

namespace {

namespace lb = laplace::bijector;
constexpr int max_n = 21;          // packed size of p = 6
constexpr int calls = 20000;  // 2000 for matrix_log, whose scratch is larger

// quantities: 0 map, 1 log-Jacobian
template <class B, int K, int Dir, int Q>
struct Bench {
  B b;
  double v0[max_n];  // the forward point
  KOKKOS_INLINE_FUNCTION double operator()(double x, double* w) const {
    double in[max_n], out[max_n];
    int idx[K > 0 ? K : 1] = {};
    const int n = Dir == 0 ? b.free_size() : b.value_size();
    if constexpr (Dir == 0)
      for (int k = 0; k < n; ++k) in[k] = x + 0.1 * k;
    else
      for (int k = 0; k < n; ++k) in[k] = v0[k];
    if constexpr (Dir == 1) {
      if constexpr (std::is_same_v<B, lb::alr_bijector>) {
        in[0] += 1e-3 * x * in[0];
        in[n - 1] -= 1e-3 * x * v0[0];
      } else if constexpr (std::is_same_v<B, lb::log_cholesky_bijector> || std::is_same_v<B, lb::matrix_log_bijector>) {
        for (int i = 0; i < b.dim; ++i) in[i * (i + 3) / 2] += 1e-3 * x;
      } else {
        for (int k = 0; k < n; ++k) in[k] *= 1.0 + 1e-3 * x;
        for (int i = 0; i < b.dim; ++i) in[i * (i + 3) / 2] = 1.0;
      }
    }
    if constexpr (Q == 1) {
      if constexpr (K == 0) return Dir == 0 ? b.inverse_log_jacobian(in, w) : b.forward_log_jacobian(in, w);
      else return Dir == 0 ? b.template inverse_log_jacobian_dk<K>(in, idx, w) : b.template forward_log_jacobian_dk<K>(in, idx, w);
    } else {
      const int m = Dir == 0 ? b.value_size() : b.free_size();
      for (int k = 0; k < m; ++k) out[k] = 0.0;
      if constexpr (K == 0) {
        if constexpr (Dir == 0) b.inverse(in, out, w);
        else b.forward(in, out, w);
      } else {
        if constexpr (Dir == 0) b.template inverse_dk<K>(in, idx, out, w);
        else b.template forward_dk<K>(in, idx, out, w);
      }
      double s = 0.0;
      for (int k = 0; k < m; ++k) s += out[k];
      return s;
    }
  }
};

// best of reps, ns per call
template <class Space, class F>
double time_calls(const F& f, const std::vector<double>& xs, int scratch, int reps) {
  using Mem = typename Space::memory_space;
  Kokkos::View<double*, Mem> x("x", xs.size());
  Kokkos::View<double**, Kokkos::LayoutRight, Mem> w("w", xs.size(), scratch);
  auto hx = Kokkos::create_mirror_view(x);
  for (std::size_t i = 0; i < xs.size(); ++i) hx(i) = xs[i];
  Kokkos::deep_copy(x, hx);
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    double sum = 0.0;
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_reduce(
        "time", Kokkos::RangePolicy<Space>(0, xs.size()),
        KOKKOS_LAMBDA(const std::size_t i, double& acc) { acc += f(x(i), &w(i, 0)); }, sum);
    Kokkos::fence();
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (sum == 12345.678) std::printf(" ");  // keep the sum alive
  }
  return best * 1e9 / double(xs.size());
}

template <class B, int K, int Dir, int Q>
void one(const char* name, B b, const double* v0, const std::vector<double>& xs) {
  Bench<B, K, Dir, Q> f{b, {}};
  for (int k = 0; k < max_n; ++k) f.v0[k] = v0[k];
  const double h = time_calls<bench::HostOne>(f, xs, b.scratch_size(), 3);
  const double d = time_calls<bench::Exec>(f, xs, b.scratch_size(), 5);
  std::printf("%-13s dim=%d %-8s %-12s K=%d  %10.1f ns (1 thread)  %9.2f ns (%s)\n", name, b.dim,
              Dir == 0 ? "inverse" : "forward", Q == 0 ? "map" : "log_jacobian", K, h, d, bench::Exec::name());
}

template <class B>
void all(const char* name, B b, const std::vector<double>& xs) {
  // the forward point: h(u) at u_k = 0.3 + 0.1 k
  double u[max_n], v0[max_n] = {};
  std::vector<double> w(b.scratch_size());
  for (int k = 0; k < b.free_size(); ++k) u[k] = 0.3 + 0.1 * k;
  b.inverse(u, v0, w.data());
  auto go = [&](auto dirc) {
    constexpr int D = decltype(dirc)::value;
    one<B, 0, D, 0>(name, b, v0, xs);
    one<B, 1, D, 0>(name, b, v0, xs);
    one<B, 3, D, 0>(name, b, v0, xs);
    one<B, 6, D, 0>(name, b, v0, xs);
    one<B, 0, D, 1>(name, b, v0, xs);
    one<B, 6, D, 1>(name, b, v0, xs);
  };
  go(std::integral_constant<int, 0>{});
  go(std::integral_constant<int, 1>{});
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<double> xs(calls);
    for (double& x : xs) x = u(rng);
    for (int K : {3, 10}) all("alr", lb::alr_bijector(K), xs);
    for (int p : {3, 6}) all("log_cholesky", lb::log_cholesky_bijector(p), xs);
    for (int p : {3, 6}) all("partial_corr", lb::partial_corr_bijector(p), xs);
    const std::vector<double> few(xs.begin(), xs.begin() + calls / 10);
    for (int p : {3, 4}) all("matrix_log", lb::matrix_log_bijector(p), few);
  }
  Kokkos::finalize();
}
