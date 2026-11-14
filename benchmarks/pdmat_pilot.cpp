// Speed of the pilot structured matrices (include/laplace/pdmat/): ns per call
// of each quantity (M, M^-1, L, log det) at orders 0, 1, 3 and 6, for p = 4
// and p = 16, theta uniform in [-3, 3], on one host thread, on the default
// execution space (OpenMP or CUDA). One call fills the whole packed matrix.
#include "measure.hpp"

#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/compound_symmetry.hpp>
#include <laplace/pdmat/diagonal.hpp>

#include <cstdio>
#include <random>

namespace {

namespace lp = laplace::pdmat;
constexpr int max_p = 16;

template <int K, class S>
KOKKOS_INLINE_FUNCTION void eval_k(const S& s, int q, const double* th, double* out) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = 0;
  if (q == 0) s.template matrix_dk<K>(th, idx, out, nullptr);
  else if (q == 1) s.template inverse_dk<K>(th, idx, out, nullptr);
  else if (q == 2) s.template cholesky_dk<K>(th, idx, out, nullptr);
  else out[0] = s.template log_det_dk<K>(th, idx, nullptr);
}

template <class S, int K>
struct Bench {
  S s;
  int q;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    double th[max_p], out[lp::packed_size(max_p)];
    for (int k = 0; k < max_p; ++k) th[k] = x;
    if constexpr (K == 0) {
      if (q == 0) s.matrix(th, out, nullptr);
      else if (q == 1) s.inverse(th, out, nullptr);
      else if (q == 2) s.cholesky(th, out, nullptr);
      else out[0] = s.log_det(th, nullptr);
    } else {
      eval_k<K>(s, q, th, out);
    }
    const int n = q == 3 ? 1 : lp::packed_size(s.dim);
    double sum = 0.0;
    for (int k = 0; k < n; ++k) sum += out[k];
    return sum;
  }
};

template <class S, int K>
void row(const char* name, int p, const bench::TimeArgs& t) {
  const char* qn[] = {"matrix", "inverse", "cholesky", "log_det"};
  for (int q = 0; q < 4; ++q) {
    const Bench<S, K> b{S(p), q};
    const double one = bench::time_ns<bench::HostOne>(t.host, b, 3);
    const double par = bench::time_ns<bench::Exec>(t.dev, b, 5);
    std::printf("%-18s p=%2d %-8s K=%d  %9.1f ns (1 thread)  %8.2f ns (%s)\n", name, p, qn[q], K, one, par,
                bench::Exec::name());
  }
}

template <class S>
void all(const char* name, const bench::TimeArgs& t) {
  for (int p : {4, 16}) {
    row<S, 0>(name, p, t);
    row<S, 1>(name, p, t);
    row<S, 3>(name, p, t);
    row<S, 6>(name, p, t);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    const auto t = bench::make_time_args(200000, [&] { return u(rng); });
    all<lp::diagonal_pdmat>("diagonal", t);
    all<lp::compound_symmetry_pdmat>("compound_symmetry", t);
    all<lp::ar1_pdmat>("ar1", t);
  }
  Kokkos::finalize();
}
