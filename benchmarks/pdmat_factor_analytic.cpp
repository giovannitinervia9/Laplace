// Speed of factor_analytic_pdmat (include/laplace/pdmat/factor_analytic.hpp):
// ns per call of M^-1 at orders 0, 1, 3, 6 by the structure (Woodbury at
// order 0, Cholesky at d = p; the recursion on the augmented matrix above)
// against the recursion on M (a Cholesky factorization of M, then X[S] =
// -X[0] sum M[T] X[S \ T] on the p x p derivatives of M given by matrix_dK),
// along the pure tuple (0, ..., 0) and along K distinct parameters spread over
// the rows, and of log det at orders 0 and 6, for (p, d) in {4, 8, 16} x
// {1, 2, 4} and (8, 8), at theta_k = 0.1 x + 0.05 sin k with x uniform in [-3, 3]; on one
// host thread (Serial) and on the default execution space (OpenMP or CUDA).
// Each call has its own row of a scratch View in the space's memory.
#include "measure.hpp"

#include <laplace/pdmat/factor_analytic.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
using fa_pdmat = lp::factor_analytic_pdmat;
constexpr int max_n = 16 * 4 + 16;  // parameters at (16, 4) and below
constexpr int max_np = 16 * 17 / 2;

// scratch of the recursion on M: M[T] and X[T] full, then the structure's
template <int K>
KOKKOS_INLINE_FUNCTION int generic_scratch(const fa_pdmat& s) {
  return 2 * (1 << K) * s.dim * s.dim + s.dim * s.dim + s.scratch_size();
}

template <int k>
KOKKOS_INLINE_FUNCTION bool matrix_of(const fa_pdmat& s, const double* t, const int* sub, double* m, double* w) {
  int idx[k];
  for (int q = 0; q < k; ++q) idx[q] = sub[q];
  return s.template matrix_dk<k>(t, idx, m, w);
}
KOKKOS_INLINE_FUNCTION bool matrix_sub(const fa_pdmat& s, const double* t, const int* sub, int k, double* m, double* w) {
  switch (k) {
    case 1: return matrix_of<1>(s, t, sub, m, w);
    case 2: return matrix_of<2>(s, t, sub, m, w);
    case 3: return matrix_of<3>(s, t, sub, m, w);
    case 4: return matrix_of<4>(s, t, sub, m, w);
    case 5: return matrix_of<5>(s, t, sub, m, w);
    default: return matrix_of<6>(s, t, sub, m, w);
  }
}

// the recursion on M to d^K M^-1 (K = 0: M^-1 by Cholesky); returns the sum of
// the packed lower triangle
template <int K>
KOKKOS_INLINE_FUNCTION double generic_inverse(const fa_pdmat& s, const double* t, const int* idx, double* w) {
  constexpr int NM = 1 << K;
  const int p = s.dim, pp = p * p;
  double* Md = w;
  double* X = Md + NM * pp;
  double* L = X + NM * pp;
  double* ws = L + pp;
  bool mz[NM], xz[NM];
  double packed[max_np];
  s.matrix(t, packed, ws);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) Md[i * p + j] = Md[j * p + i] = packed[lp::packed_index(i, j)];
  for (int T = 1; T < NM; ++T) {
    int sub[6], k = 0;
    for (int q = 0; q < K; ++q)
      if (T & (1 << q)) sub[k++] = idx[q];
    mz[T] = matrix_sub(s, t, sub, k, packed, ws);
    if (mz[T])
      for (int i = 0; i < p; ++i)
        for (int j = 0; j <= i; ++j) Md[T * pp + i * p + j] = Md[T * pp + j * p + i] = packed[lp::packed_index(i, j)];
  }
  // X[0] = M^-1 by Cholesky: L, then solve for each column
  for (int j = 0; j < p; ++j) {
    double d = Md[j * p + j];
    for (int k = 0; k < j; ++k) d -= L[j * p + k] * L[j * p + k];
    L[j * p + j] = Kokkos::sqrt(d);
    for (int i = j + 1; i < p; ++i) {
      double r = Md[i * p + j];
      for (int k = 0; k < j; ++k) r -= L[i * p + k] * L[j * p + k];
      L[i * p + j] = r / L[j * p + j];
    }
  }
  for (int c = 0; c < p; ++c) {
    for (int i = 0; i < p; ++i) {
      double r = i == c ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k) r -= L[i * p + k] * X[k * p + c];
      X[i * p + c] = r / L[i * p + i];
    }
    for (int i = p - 1; i >= 0; --i) {
      double r = X[i * p + c];
      for (int k = i + 1; k < p; ++k) r -= L[k * p + i] * X[k * p + c];
      X[i * p + c] = r / L[i * p + i];
    }
  }
  xz[0] = true;
  for (int Q = 1; Q < NM; ++Q) {
    double* acc = L;  // reused
    for (int e = 0; e < pp; ++e) acc[e] = 0.0;
    xz[Q] = false;
    for (int T = Q; T != 0; T = (T - 1) & Q) {
      if (!mz[T] || !xz[Q ^ T]) continue;
      xz[Q] = true;
      const double* A = Md + T * pp;
      const double* B = X + (Q ^ T) * pp;
      for (int i = 0; i < p; ++i)
        for (int k = 0; k < p; ++k) {
          const double a = A[i * p + k];
          if (a == 0.0) continue;
          for (int j = 0; j < p; ++j) acc[i * p + j] += a * B[k * p + j];
        }
    }
    if (!xz[Q]) continue;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j < p; ++j) {
        double r = 0.0;
        for (int k = 0; k < p; ++k) r += X[i * p + k] * acc[k * p + j];
        X[Q * pp + i * p + j] = -r;
      }
  }
  double sum = 0.0;
  if (xz[NM - 1])
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) sum += X[(NM - 1) * pp + i * p + j];
  return sum;
}

// route: 0 the structure, 1 the recursion on M; q: 0 M^-1, 1 log det
template <int K>
struct Bench {
  fa_pdmat s;
  int route, q;
  bool mixed;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double* w) const {
    double u[max_n], out[max_np] = {};
    int idx[K > 0 ? K : 1] = {};
    const int n = s.n_params();
    for (int k = 0; k < n; ++k) u[k] = 0.1 * x + 0.05 * Kokkos::sin(double(k));
    if (mixed)  // K parameters spread over the rows: idx_k = k (n - 1)/K
      for (int k = 0; k < K; ++k) idx[k] = k * (n - 1) / K;
    if (route == 1) return generic_inverse<K>(s, u, idx, w);
    if constexpr (K == 0) {
      if (q == 1) return s.log_det(u, w);
      s.inverse(u, out, w);
    } else {
      if (q == 1) return s.template log_det_dk<K>(u, idx, w);
      s.template inverse_dk<K>(u, idx, out, w);
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
void one(int p, int d, int route, int q, bool mixed, const std::vector<double>& all) {
  const fa_pdmat s(p, d);
  const int sw = route == 1 ? generic_scratch<K>(s) : s.scratch_size();
  // at most about 200 MB of scratch
  const std::size_t n = std::min<std::size_t>(all.size(), std::size_t(2.5e7 / sw) + 1);
  const std::vector<double> xs(all.begin(), all.begin() + n);
  const Bench<K> f{s, route, q, mixed};
  const double h = time_calls<bench::HostOne>(f, xs, sw, 3);
  const double e = time_calls<bench::Exec>(f, xs, sw, 5);
  std::printf("p=%2d d=%d %-8s %-8s K=%d %-5s %11.1f ns (1 thread)  %10.2f ns (%s)\n", p, d,
              route == 0 ? "struct" : "M route", q == 0 ? "inverse" : "log_det", K,
              K == 0 ? "" : (mixed ? "mixed" : "pure"), h, e, bench::Exec::name());
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
        for (int route : {0, 1}) {
          one<0>(p, d, route, 0, false, xs);
          for (bool mixed : {false, true}) {
            one<1>(p, d, route, 0, mixed, xs);
            one<3>(p, d, route, 0, mixed, xs);
            one<6>(p, d, route, 0, mixed, xs);
          }
        }
        one<0>(p, d, 0, 1, false, xs);
        one<6>(p, d, 0, 1, true, xs);
      }
    // log det at d = p (the traces on M) for a larger p
    one<0>(8, 8, 0, 1, false, xs);
    for (bool mixed : {false, true}) {
      one<3>(8, 8, 0, 1, mixed, xs);
      one<6>(8, 8, 0, 1, mixed, xs);
    }
  }
  Kokkos::finalize();
}
