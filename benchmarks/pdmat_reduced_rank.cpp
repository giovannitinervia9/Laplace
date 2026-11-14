// Speed of reduced_rank_psdmat (include/laplace/pdmat/reduced_rank.hpp): ns
// per call of M at orders 0 and 1 and of log|M|_+ at orders 0, 1, 3, 6 (the
// recursion on Lambda^T Lambda), along K distinct parameters spread over the
// rows, for (p, d) in {4, 8, 16} x {1, 2, 4}, at theta_k = 0.1 x + 0.05 sin k
// with x uniform in [-3, 3]; on one host thread (Serial) and on the default
// execution space (OpenMP or CUDA). Each call has its own row of a scratch
// View in the space's memory.
#include "measure.hpp"

#include <laplace/pdmat/reduced_rank.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
using rr_psdmat = lp::reduced_rank_psdmat;
constexpr int max_n = 16 * 4;  // parameters at (16, 4) and below
constexpr int max_np = 16 * 17 / 2;

// q: 0 M, 1 log|M|_+
template <int K>
struct Bench {
  rr_psdmat s;
  int q;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double* w) const {
    double u[max_n], out[max_np] = {};
    int idx[K > 0 ? K : 1] = {};
    const int n = s.n_params();
    for (int k = 0; k < n; ++k) u[k] = 0.1 * x + 0.05 * Kokkos::sin(double(k));
    for (int k = 0; k < K; ++k) idx[k] = k * (n - 1) / K;  // spread over the rows
    if constexpr (K == 0) {
      if (q == 1) return s.log_pdet(u, w);
      s.matrix(u, out, w);
    } else {
      if (q == 1) return s.template log_pdet_dk<K>(u, idx, w);
      s.template matrix_dk<K>(u, idx, out, w);
    }
    double sum = 0.0;
    for (int k = 0; k < lp::packed_size(s.dim); ++k) sum += out[k];
    return sum;
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

template <int K>
void one(int p, int d, int q, const std::vector<double>& all) {
  const rr_psdmat s(p, d);
  const int sw = s.scratch_size();
  const std::size_t n = std::min<std::size_t>(all.size(), std::size_t(2.5e7 / sw) + 1);
  const std::vector<double> xs(all.begin(), all.begin() + n);
  const Bench<K> f{s, q};
  const double h = time_calls<bench::HostOne>(f, xs, sw, 3);
  const double e = time_calls<bench::Exec>(f, xs, sw, 5);
  std::printf("p=%2d d=%d %-8s K=%d %11.1f ns (1 thread)  %10.2f ns (%s)\n", p, d, q == 0 ? "matrix" : "log_pdet", K, h,
              e, bench::Exec::name());
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<double> xs(20000);
    for (double& x : xs) x = u(rng);
    for (int p : {4, 8, 16})
      for (int d : {1, 2, 4}) {
        if (d > p) continue;
        one<0>(p, d, 0, xs);
        one<1>(p, d, 0, xs);
        one<0>(p, d, 1, xs);
        one<1>(p, d, 1, xs);
        one<3>(p, d, 1, xs);
        one<6>(p, d, 1, xs);
      }
  }
  Kokkos::finalize();
}
