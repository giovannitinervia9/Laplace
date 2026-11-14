// Speed of toeplitz_pdmat (include/laplace/pdmat/toeplitz.hpp): ns per call of
// M, M^-1 and L at orders 0, 1, 3, 6 and of log det at orders 0 and 6, for
// p = 3, 6, 10, along three tuples: the pure tuple of the first parameter
// (0, ..., 0), which no stage of the recursion can skip, the pure tuple of the
// last parameter (q-1, ..., q-1), whose masks vanish until the last stage, and
// the spread tuple idx[k] = k q / K; at theta_k = 0.1 x + 0.05 k with x uniform
// in [-3, 3]; on one host thread (Serial) and on the default execution space
// (OpenMP or CUDA), each call with its own row of a scratch View.
#include "measure.hpp"

#include <laplace/pdmat/toeplitz.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
constexpr int max_n = 9;  // parameters at p = 10
constexpr int max_out = 55;
constexpr int calls = 20000;

// q: 0 M, 1 M^-1, 2 L, 3 log det; tuple: 0 first, 1 last, 2 spread
template <int K>
struct Bench {
  lp::toeplitz_pdmat s;
  int q, tuple;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double* w) const {
    double t[max_n], out[max_out] = {};
    int idx[K > 0 ? K : 1] = {};
    const int n = s.n_params();
    for (int k = 0; k < max_n; ++k) t[k] = 0.1 * x + 0.05 * k;
    if constexpr (K > 0)
      for (int k = 0; k < K; ++k) idx[k] = tuple == 0 ? 0 : tuple == 1 ? n - 1 : (k * n) / K;
    if constexpr (K == 0) {
      if (q == 0) s.matrix(t, out, w);
      else if (q == 1) s.inverse(t, out, w);
      else if (q == 2) s.cholesky(t, out, w);
      else return s.log_det(t, w);
    } else {
      if (q == 0) s.template matrix_dk<K>(t, idx, out, w);
      else if (q == 1) s.template inverse_dk<K>(t, idx, out, w);
      else if (q == 2) s.template cholesky_dk<K>(t, idx, out, w);
      else return s.template log_det_dk<K>(t, idx, w);
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
  Kokkos::View<double**, Kokkos::LayoutRight, Mem> w("w", xs.size(), std::max(1, scratch));
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

const char* qname[] = {"matrix", "inverse", "cholesky", "log_det"};
const char* tname[] = {"first", "last", "spread"};

template <int K>
void one(int p, int q, int tuple, const std::vector<double>& xs) {
  const lp::toeplitz_pdmat s(p);
  const Bench<K> f{s, q, tuple};
  const double h = time_calls<bench::HostOne>(f, xs, s.scratch_size(), 3);
  const double d = time_calls<bench::Exec>(f, xs, s.scratch_size(), 5);
  std::printf("toeplitz p=%-2d %-8s K=%d %-6s %10.1f ns (1 thread)  %9.2f ns (%s)\n", p, qname[q], K, K == 0 ? "" : tname[tuple],
              h, d, bench::Exec::name());
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<double> xs(calls);
    for (double& x : xs) x = u(rng);
    for (int p : {3, 6, 10}) {
      const std::vector<double> few(xs.begin(), xs.begin() + (p == 10 ? calls / 4 : calls));
      for (int q = 0; q < 3; ++q) {
        one<0>(p, q, 0, few);
        for (int tuple = 0; tuple < 3; ++tuple) {
          one<1>(p, q, tuple, few);
          one<3>(p, q, tuple, few);
          one<6>(p, q, tuple, few);
        }
      }
      one<0>(p, 3, 0, few);
      one<6>(p, 3, 0, few);
    }
  }
  Kokkos::finalize();
}
