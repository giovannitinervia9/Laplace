// Speed of the B-spline basis (include/laplace/basis/bspline.hpp): the rows
// of n points, cubic, uniform clamped knots on [0, 1], K = 10, 40, 200, for
// the values, the second derivative and the integral, on one host thread
// (Kokkos::Serial) and on the default execution space (OpenMP or CUDA).
// Two outputs: the sparse rows of Laplace (first column + d + 1 values) and a
// dense n x K matrix (zeros, the window, and for the integral the column
// totals before it), which is what R's splines2 returns; the dense one is
// skipped when it exceeds 512 MB. Median of the repetitions, ms per call.
// Usage: basis_bspline [n ...]   (default 100000 1000000)
#include <laplace/basis/bspline.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace lb = laplace::basis;

template <class Dev>
struct dense_row_functor {
  lb::bspline_basis b;
  int k;
  Kokkos::View<double*, Dev> x;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> out;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double r[lb::bspline_max_degree + 1];
    const int f = lb::basis_row_of_order(b, k, x(i), r);
    for (int j = 0; j < b.dim; ++j) out(i, j) = 0.0;
    for (int q = 0; q <= b.degree; ++q) out(i, f + q) = r[q];
  }
};

template <class Dev>
struct sparse_integral_functor {
  lb::bspline_basis b;
  Kokkos::View<double*, Dev> x;
  Kokkos::View<int*, Dev> first;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> out;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const { first(i) = b.integral_row(x(i), &out(i, 0)); }
};

template <class Dev>
struct dense_integral_functor {
  lb::bspline_basis b;
  Kokkos::View<double*, Dev> x, total;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> out;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double r[lb::bspline_max_degree + 1];
    const int f = b.integral_row(x(i), r);
    for (int j = 0; j < f; ++j) out(i, j) = total(j);
    for (int q = 0; q <= b.degree; ++q) out(i, f + q) = r[q];
    for (int j = f + b.degree + 1; j < b.dim; ++j) out(i, j) = 0.0;
  }
};

template <class Exec, class Run>
double median_ms(Run run, int reps) {
  run();
  Exec().fence();
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    Kokkos::Timer tm;
    run();
    Exec().fence();
    t.push_back(tm.seconds() * 1e3);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

template <class Exec>
void run_space(const char* label, const std::vector<double>& xh, int reps) {
  using Dev = Kokkos::Device<Exec, typename Exec::memory_space>;
  const int n = static_cast<int>(xh.size());
  Kokkos::View<double*, Dev> x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) hx(i) = xh[i];
  Kokkos::deep_copy(x, hx);
  for (int K : {10, 40, 200}) {
    const int d = 3;
    auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, K, d);
    const bool on_host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;
    const auto b = on_host ? owner.host() : owner.device();
    Kokkos::View<int*, Dev> first("first", n);
    Kokkos::View<double**, Kokkos::LayoutRight, Dev> sv("sparse", n, d + 1);
    const bool dense_ok = static_cast<double>(n) * K * 8 <= 512e6;
    Kokkos::View<double**, Kokkos::LayoutRight, Dev> dv("dense", dense_ok ? n : 0, K);
    Kokkos::View<double*, Dev> total("total", K);
    auto ht = Kokkos::create_mirror_view(total);
    const auto hb = owner.host();
    for (int j = 0; j < K; ++j) ht(j) = hb.integral_column_total(j);
    Kokkos::deep_copy(total, ht);
    for (int k : {0, 2, -1}) {  // -1: the integral
      double ts, td = -1.0;
      if (k >= 0) {
        ts = median_ms<Exec>([&] { lb::fill_basis_rows(b, k, x, first, sv); }, reps);
        if (dense_ok)
          td = median_ms<Exec>([&] {
            Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), dense_row_functor<Dev>{b, k, x, dv});
          }, reps);
      } else {
        ts = median_ms<Exec>([&] {
          Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), sparse_integral_functor<Dev>{b, x, first, sv});
        }, reps);
        if (dense_ok)
          td = median_ms<Exec>([&] {
            Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), dense_integral_functor<Dev>{b, x, total, dv});
          }, reps);
      }
      const char* what = k == 0 ? "values" : k == 2 ? "deriv2" : "integral";
      if (td >= 0)
        std::printf("%-7s n=%8d K=%3d %-8s sparse %9.3f ms  dense %9.3f ms\n", label, n, K, what, ts, td);
      else
        std::printf("%-7s n=%8d K=%3d %-8s sparse %9.3f ms  dense      (skipped)\n", label, n, K, what, ts);
    }
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::vector<int> ns;
    for (int a = 1; a < argc; ++a)
      if (argv[a][0] != '-') ns.push_back(std::atoi(argv[a]));
    if (ns.empty()) ns = {100000, 1000000};
    for (int n : ns) {
      std::mt19937_64 gen(20261114);
      std::uniform_real_distribution<double> U(0.0, 1.0);
      std::vector<double> xh(n);
      for (auto& v : xh) v = U(gen);
#ifdef KOKKOS_ENABLE_SERIAL
      run_space<Kokkos::Serial>("serial", xh, 5);
#endif
      if (!std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>)
        run_space<Kokkos::DefaultExecutionSpace>(Kokkos::DefaultExecutionSpace::name(), xh, 21);
    }
  }
  Kokkos::finalize();
  return 0;
}
