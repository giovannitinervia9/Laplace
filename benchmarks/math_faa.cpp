// Accuracy and speed of Faa di Bruno's formula to order 7
// (include/laplace/math/faa_di_bruno.hpp) against the mpmath references.
//
// Usage: math_faa <reference dir>
#include "faa_common.hpp"

#include <random>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_faa <reference dir>\n");
    return 2;
  }
  bench::print_spaces();
  const auto r = faa::read(std::string(argv[1]) + "/faa_di_bruno.csv");
  // timing inputs: 2^20 random rows
  const std::size_t m = std::size_t(1) << 20;
  using V = Kokkos::View<double* [16], Kokkos::LayoutRight, bench::Exec::memory_space>;
  V d("t", m);
  auto h = Kokkos::create_mirror_view(d);
  std::mt19937_64 gen(1);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  for (std::size_t i = 0; i < m; ++i)
    for (int c = 0; c < 16; ++c) h(i, c) = u(gen);
  Kokkos::deep_copy(d, h);
  Kokkos::View<double*, bench::Exec::memory_space> y("y", m);
  for (int n = 1; n <= 7; ++n) {
    const auto e = faa::errors(r, faa::eval(r, n), n);
    double ns = 1e300;
    for (int rep = 0; rep < 5; ++rep) {
      Kokkos::fence();
      const auto t0 = std::chrono::steady_clock::now();
      Kokkos::parallel_for(
          "faa_t", Kokkos::RangePolicy<bench::Exec>(0, m), KOKKOS_LAMBDA(const std::size_t i) {
            double f[8], g[8];
            for (int c = 0; c < 8; ++c) f[c] = d(i, c), g[c] = d(i, 8 + c);
            y(i) = faa::compose(n, f, g);
          });
      Kokkos::fence();
      ns = std::min(ns, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / m);
    }
    std::printf("chain_rule_d%d | %s: max %.3g ulp, %.3g ulpS (row %zu) | non-finite %zu | ns/call %.3f\n", n,
                bench::Exec::name(), e.ulp, e.ulpS, e.worst, e.bad, ns);
  }
  return 0;
}
