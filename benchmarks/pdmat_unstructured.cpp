// Speed of unstructured_pdmat<B> (include/laplace/pdmat/unstructured.hpp):
// ns per call of M^-1 at orders 0, 1, 3, 6, along the pure tuple (0, ..., 0)
// and along the tuple of K distinct coordinates (0, 1, ..., K-1), and of
// log det at orders 0 and 6, for p = 3 and 4, at u_k = 0.1 x + 0.05 k with x
// uniform in [-3, 3]; on one host thread (Serial) and on the default execution
// space (OpenMP or CUDA). Each call has its own row of a scratch View in the
// space's memory, as a caller on the device would provide. M, L and the
// log-Jacobian are the bijector's (benchmarks/bijector_coupled.cpp).
// With the argument "merge": M^-1 at orders 4-6 along pure, grouped
// (0, ..., 0, 1, ..., 1) and distinct tuples, p = 3, 5, 10 (merge_grid).
#include "measure.hpp"

#include <laplace/pdmat/unstructured.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
namespace lb = laplace::bijector;
constexpr int max_n = 55;  // free size at p = 10
constexpr int calls = 20000;

// q: 0 M^-1, 1 log det; mixed: the tuple (0, 1, ..., K-1), else (0, ..., 0);
// grouped (mixed ignored): two groups, (0, ..., 0, 1, ..., 1) with K/2 ones
template <class S, int K>
struct Bench {
  S s;
  int q;
  bool mixed;
  bool grouped = false;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double* w) const {
    double u[max_n], out[max_n] = {};
    int idx[K > 0 ? K : 1] = {};
    for (int k = 0; k < max_n; ++k) u[k] = 0.1 * x + 0.05 * k;
    if (grouped)
      for (int k = 0; k < K; ++k) idx[k] = k >= K - K / 2 ? 1 : 0;
    else if (mixed)
      for (int k = 0; k < K; ++k) idx[k] = k;
    if constexpr (K == 0) {
      if (q == 0) s.inverse(u, out, w);
      else return s.log_det(u, w);
    } else {
      if (q == 0) s.template inverse_dk<K>(u, idx, out, w);
      else return s.template log_det_dk<K>(u, idx, w);
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

template <class S, int K>
void one(const char* name, int p, int q, bool mixed, const std::vector<double>& xs) {
  const S s(p);
  if (mixed && K > s.n_params()) return;
  const Bench<S, K> f{s, q, mixed};
  const double h = time_calls<bench::HostOne>(f, xs, s.scratch_size(), 3);
  const double d = time_calls<bench::Exec>(f, xs, s.scratch_size(), 5);
  std::printf("%-13s p=%d %-8s K=%d %-5s %10.1f ns (1 thread)  %9.2f ns (%s)\n", name, p, q == 0 ? "inverse" : "log_det",
              K, K == 0 ? "" : (mixed ? "mixed" : "pure"), h, d, bench::Exec::name());
}

// the grid of the sub-tuple merging: orders 4-6, pure / grouped / distinct
// tuples, p = 3, 5, 10, M^-1 only; fewer calls as p grows
template <class S, int K>
void merge_one(const char* name, int p, int tuple, const std::vector<double>& xs) {
  const S s(p);
  if (tuple == 2 && K > s.n_params()) return;
  const int n = p == 3 ? 4000 : p == 5 ? 1000 : 100;
  const std::vector<double> few(xs.begin(), xs.begin() + n);
  const Bench<S, K> f{s, 0, tuple == 2, tuple == 1};
  const double h = time_calls<bench::HostOne>(f, few, s.scratch_size(), 3);
  const double d = time_calls<bench::Exec>(f, few, s.scratch_size(), 5);
  std::printf("%-13s p=%-2d K=%d %-8s %12.1f ns (1 thread) %11.1f ns (%s)\n", name, p, K,
              tuple == 0 ? "pure" : tuple == 1 ? "grouped" : "distinct", h, d, bench::Exec::name());
}

template <class S>
void merge_grid(const char* name, const std::vector<double>& xs) {
  for (int p : {3, 5, 10})
    for (int tuple : {0, 1, 2}) {
      merge_one<S, 4>(name, p, tuple, xs);
      merge_one<S, 5>(name, p, tuple, xs);
      merge_one<S, 6>(name, p, tuple, xs);
    }
}

template <class S>
void all(const char* name, const std::vector<double>& xs) {
  for (int p : {3, 4}) {
    one<S, 0>(name, p, 0, false, xs);
    for (bool mixed : {false, true}) {
      one<S, 1>(name, p, 0, mixed, xs);
      one<S, 3>(name, p, 0, mixed, xs);
      one<S, 6>(name, p, 0, mixed, xs);
    }
    one<S, 0>(name, p, 1, false, xs);
    one<S, 6>(name, p, 1, false, xs);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<double> xs(calls);
    for (double& x : xs) x = u(rng);
    if (argc > 1 && std::string(argv[1]) == "merge") {
      merge_grid<lp::unstructured_pdmat<lb::log_cholesky_bijector>>("log_cholesky", xs);
      merge_grid<lp::unstructured_corr_pdmat>("partial_corr", xs);
      merge_grid<lp::unstructured_pdmat<lb::matrix_log_bijector>>("matrix_log", xs);
      Kokkos::finalize();
      return 0;
    }
    all<lp::unstructured_pdmat<lb::log_cholesky_bijector>>("log_cholesky", xs);
    all<lp::unstructured_corr_pdmat>("partial_corr", xs);
    // matrix_log: about 25000 doubles of scratch a call, so fewer calls
    const std::vector<double> few(xs.begin(), xs.begin() + calls / 10);
    all<lp::unstructured_pdmat<lb::matrix_log_bijector>>("matrix_log", few);
  }
  Kokkos::finalize();
}
