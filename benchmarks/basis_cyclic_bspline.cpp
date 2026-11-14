// Speed of the cyclic B-spline basis (include/laplace/basis/cyclic_bspline.hpp)
// against the ordinary B-spline basis with the same knots inside the domain
// (uniform_extended, K = 10, 40, 200, cubic, [0, 1)): the sparse rows of n
// points for the values, the second derivative and the integral, on one host
// thread (Kokkos::Serial) and on the default execution space (OpenMP or
// CUDA). The cyclic rows cost the reduction modulo the period (fmod) on top;
// its integral row also adds the column totals of the finished pieces.
// Points uniform on [-3, 4) (several periods, every row reduced) and on
// [0, 1) (one period, no reduction). Median of the repetitions, ms per call.
// Usage: basis_cyclic_bspline [n ...]   (default 1000000)
#include <laplace/basis/cyclic_bspline.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace lb = laplace::basis;

template <class B, class Dev>
struct integral_functor {
  B b;
  Kokkos::View<double*, Dev> x;
  Kokkos::View<int*, Dev> first;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> out;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const { first(i) = b.integral_row(x(i), &out(i, 0)); }
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

template <class Exec, class B>
double time_basis(const B& b, int k, const Kokkos::View<double*, Kokkos::Device<Exec, typename Exec::memory_space>>& x,
                  int reps) {
  using Dev = Kokkos::Device<Exec, typename Exec::memory_space>;
  const int n = static_cast<int>(x.extent(0));
  Kokkos::View<int*, Dev> first("first", n);
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> sv("sparse", n, b.row_width());
  if (k >= 0) return median_ms<Exec>([&] { lb::fill_basis_rows(b, k, x, first, sv); }, reps);
  return median_ms<Exec>([&] {
    Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), integral_functor<B, Dev>{b, x, first, sv});
  }, reps);
}

template <class Exec>
void run_space(const char* label, const std::vector<double>& xh, int reps, const char* where) {
  using Dev = Kokkos::Device<Exec, typename Exec::memory_space>;
  const int n = static_cast<int>(xh.size());
  Kokkos::View<double*, Dev> x("x", n), xr("x reduced", n);
  auto hx = Kokkos::create_mirror_view(x);
  auto hr = Kokkos::create_mirror_view(xr);
  for (int i = 0; i < n; ++i) {
    hx(i) = xh[i];
    hr(i) = xh[i] - std::floor(xh[i]);  // the same points in [0, 1) for the ordinary basis
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(xr, hr);
  const bool on_host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;
  for (int K : {10, 40, 200}) {
    const int d = 3;
    auto co = lb::cyclic_bspline_basis_owner::uniform(0.0, 1.0, K, d);
    auto bo = lb::bspline_basis_owner::uniform_extended(0.0, 1.0, K + d, d);
    const auto cb = on_host ? co.host() : co.device();
    const auto bb = on_host ? bo.host() : bo.device();
    for (int k : {0, 2, -1}) {
      const double tc = time_basis<Exec>(cb, k, x, reps), tb = time_basis<Exec>(bb, k, xr, reps);
      const char* what = k == 0 ? "values" : k == 2 ? "deriv2" : "integral";
      std::printf("%-7s n=%8d K=%3d %-8s cyclic %9.3f ms  bspline %9.3f ms  ratio %5.2f\n", label, n, K, what, tc, tb,
                  tc / tb);
    }
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::vector<int> ns;
    for (int a = 1; a < argc; ++a)
      if (argv[a][0] != '-') ns.push_back(std::atoi(argv[a]));
    if (ns.empty()) ns = {1000000};
    for (int n : ns) {
      for (double a : {-3.0, 0.0}) {
        std::mt19937_64 gen(20261114);
        std::uniform_real_distribution<double> U(a, a == 0.0 ? 1.0 : 4.0);
        std::vector<double> xh(n);
        for (auto& v : xh) v = U(gen);
        const char* where = a == 0.0 ? "[0,1)" : "[-3,4)";
#ifdef KOKKOS_ENABLE_SERIAL
        run_space<Kokkos::Serial>("serial", xh, 5, where);
#endif
        if (!std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>)
          run_space<Kokkos::DefaultExecutionSpace>(Kokkos::DefaultExecutionSpace::name(), xh, 21, where);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
