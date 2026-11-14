// Speed and memory of the tensor product bases (include/laplace/basis/tensor.hpp)
// with cubic B-spline margins on [0, 1] (uniform_clamped), D = 2 with K_j = 10
// and 20, D = 3 with K_j = 10, n points uniform in the box, on one host thread
// (Kokkos::Serial) and on the default execution space (OpenMP or CUDA).
//   rows        the sparse rows (D first columns and prod w_j values)
//   rows fixed  D = 2 only: the margins' row() called directly and a plain
//               double loop, no multi-index (the reference for P3)
//   dense rows  the same rows scattered into a dense n x K matrix (skipped
//               above 4e7 doubles)
//   expansion   expansion() on the fly; dot from stored sparse rows; dot from
//               stored dense rows
//   integral    D = 2: the row with k = (0, -1), the second margin integrated,
//               for a B-spline and for a cyclic second margin (P5)
// Median of the repetitions, ms per call. Usage: basis_tensor [n ...]
// (default 100000).
#include <laplace/basis/bspline.hpp>
#include <laplace/basis/cyclic_bspline.hpp>
#include <laplace/basis/tensor.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace lb = laplace::basis;

constexpr int SCR = 64;  // local scratch: sum_j w_j(k_j) <= 64 in every configuration here

template <class TB, class Dev>
struct rows_functor {
  static constexpr int D = TB::margin_count;
  TB b;
  Kokkos::Array<int, D> k;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> x, out;
  Kokkos::View<int**, Kokkos::LayoutRight, Dev> first;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double xi[D], s[SCR];
    int f[D];
    for (int j = 0; j < D; ++j) xi[j] = x(i, j);
    b.row_partial(xi, k.data(), f, &out(i, 0), s);
    for (int j = 0; j < D; ++j) first(i, j) = f[j];
  }
};

// D = 2, order 0, the margins called directly
template <class TB, class Dev>
struct fixed_functor {
  TB b;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> x, out;
  Kokkos::View<int**, Kokkos::LayoutRight, Dev> first;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double r0[16], r1[16];
    const int f0 = b.template margin<0>().row(x(i, 0), r0), f1 = b.template margin<1>().row(x(i, 1), r1);
    const int w0 = b.template margin<0>().row_width(), w1 = b.template margin<1>().row_width();
    for (int a = 0; a < w0; ++a)
      for (int c = 0; c < w1; ++c) out(i, a * w1 + c) = r0[a] * r1[c];
    first(i, 0) = f0;
    first(i, 1) = f1;
  }
};

template <class TB, class Dev>
struct dense_functor {
  static constexpr int D = TB::margin_count;
  TB b;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> x, out;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double xi[D], s[SCR], r[SCR];
    int f[D];
    const Kokkos::Array<int, D> k{};
    for (int j = 0; j < D; ++j) xi[j] = x(i, j);
    const int W = b.row_partial(xi, k.data(), f, r, s);
    for (int c = 0; c < static_cast<int>(out.extent(1)); ++c) out(i, c) = 0.0;
    for (int q = 0; q < W; ++q) out(i, b.column(k.data(), f, q)) = r[q];
  }
};

template <class TB, class Dev>
struct expansion_functor {
  static constexpr int D = TB::margin_count;
  TB b;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> x;
  Kokkos::View<double*, Dev> beta, eta;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double xi[D], s[SCR];
    for (int j = 0; j < D; ++j) xi[j] = x(i, j);
    eta(i) = b.expansion(xi, beta.data(), s);
  }
};

template <class TB, class Dev>
struct sparse_dot_functor {
  static constexpr int D = TB::margin_count;
  TB b;
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> rows;
  Kokkos::View<int**, Kokkos::LayoutRight, Dev> first;
  Kokkos::View<double*, Dev> beta, eta;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    int f[D];
    for (int j = 0; j < D; ++j) f[j] = first(i, j);
    double s = 0;
    const int W = static_cast<int>(rows.extent(1));
    for (int q = 0; q < W; ++q) s += rows(i, q) * beta(b.column(f, q));
    eta(i) = s;
  }
};

template <class Dev>
struct dense_dot_functor {
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> rows;
  Kokkos::View<double*, Dev> beta, eta;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double s = 0;
    for (int c = 0; c < static_cast<int>(rows.extent(1)); ++c) s += rows(i, c) * beta(c);
    eta(i) = s;
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

template <class Exec, class TB>
void time_product(const char* label, const char* name, const TB& b, int n, int reps) {
  using Dev = Kokkos::Device<Exec, typename Exec::memory_space>;
  constexpr int D = TB::margin_count;
  const int K = b.dimension(), W = b.row_width();
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> x("x", n, D), rows("rows", n, W);
  Kokkos::View<int**, Kokkos::LayoutRight, Dev> first("first", n, D);
  Kokkos::View<double*, Dev> beta("beta", K), eta("eta", n);
  auto xh = Kokkos::create_mirror_view(x);
  auto bh = Kokkos::create_mirror_view(beta);
  std::mt19937_64 gen(20261114);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < D; ++j) xh(i, j) = U(gen);
  for (int c = 0; c < K; ++c) bh(c) = U(gen) - 0.5;
  Kokkos::deep_copy(x, xh);
  Kokkos::deep_copy(beta, bh);
  const Kokkos::RangePolicy<Exec> pol(0, n);
  const Kokkos::Array<int, D> k0{};
  const double t_rows = median_ms<Exec>([&] { Kokkos::parallel_for(pol, rows_functor<TB, Dev>{b, k0, x, rows, first}); }, reps);
  double t_fixed = -1;
  if constexpr (D == 2)
    t_fixed = median_ms<Exec>([&] { Kokkos::parallel_for(pol, fixed_functor<TB, Dev>{b, x, rows, first}); }, reps);
  Kokkos::parallel_for(pol, rows_functor<TB, Dev>{b, k0, x, rows, first});
  const double t_exp = median_ms<Exec>([&] { Kokkos::parallel_for(pol, expansion_functor<TB, Dev>{b, x, beta, eta}); }, reps);
  const double t_sdot =
      median_ms<Exec>([&] { Kokkos::parallel_for(pol, sparse_dot_functor<TB, Dev>{b, rows, first, beta, eta}); }, reps);
  double t_dense = -1, t_ddot = -1;
  if (static_cast<double>(n) * K <= 4e7) {
    Kokkos::View<double**, Kokkos::LayoutRight, Dev> dense("dense", n, K);
    t_dense = median_ms<Exec>([&] { Kokkos::parallel_for(pol, dense_functor<TB, Dev>{b, x, dense}); }, reps);
    t_ddot = median_ms<Exec>([&] { Kokkos::parallel_for(pol, dense_dot_functor<Dev>{dense, beta, eta}); }, reps);
  }
  std::printf("%-7s n=%7d %-14s K=%5d W=%3d  bytes/row sparse %4d dense %6d | rows %8.3f fixed %8.3f dense %9.3f |"
              " expansion %8.3f sparse-dot %8.3f dense-dot %9.3f ms\n",
              label, n, name, K, W, static_cast<int>(W * 8 + D * 4), K * 8, t_rows, t_fixed, t_dense, t_exp, t_sdot,
              t_ddot);
}

template <class Exec, class TB>
double time_integral(const TB& b, int n, int reps) {
  using Dev = Kokkos::Device<Exec, typename Exec::memory_space>;
  const Kokkos::Array<int, 2> k{0, -1};
  Kokkos::View<double**, Kokkos::LayoutRight, Dev> x("x", n, 2), rows("rows", n, b.row_width(k.data()));
  Kokkos::View<int**, Kokkos::LayoutRight, Dev> first("first", n, 2);
  auto xh = Kokkos::create_mirror_view(x);
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 2; ++j) xh(i, j) = U(gen);
  Kokkos::deep_copy(x, xh);
  return median_ms<Exec>(
      [&] { Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), rows_functor<TB, Dev>{b, k, x, rows, first}); }, reps);
}

template <class Exec>
void run_space(const char* label, int n, int reps) {
  const bool on_host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;
  for (int K : {10, 20}) {
    auto o = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, K, 3);
    const auto m = on_host ? o.host() : o.device();
    time_product<Exec>(label, K == 10 ? "D=2 K_j=10" : "D=2 K_j=20", lb::tensor_product_basis(m, m), n, reps);
  }
  {
    auto o = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 10, 3);
    const auto m = on_host ? o.host() : o.device();
    time_product<Exec>(label, "D=3 K_j=10", lb::tensor_product_basis(m, m, m), n, reps);
  }
  for (int K : {10, 20}) {
    auto o = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, K, 3);
    auto c = lb::cyclic_bspline_basis_owner::uniform(0.0, 1.0, K, 3);
    const auto m = on_host ? o.host() : o.device();
    const auto mc = on_host ? c.host() : c.device();
    const double tb = time_integral<Exec>(lb::tensor_product_basis(m, m), n, reps);
    const double tc = time_integral<Exec>(lb::tensor_product_basis(m, mc), n, reps);
    std::printf("%-7s n=%7d integral k=(0,-1) K_j=%2d: bspline second margin %8.3f ms, cyclic %8.3f ms, ratio %5.2f\n",
                label, n, K, tb, tc, tc / tb);
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::vector<int> ns;
    for (int a = 1; a < argc; ++a)
      if (argv[a][0] != '-') ns.push_back(std::atoi(argv[a]));
    if (ns.empty()) ns = {100000};
    for (int n : ns) {
#ifdef KOKKOS_ENABLE_SERIAL
      run_space<Kokkos::Serial>("serial", n, 5);
#endif
      if (!std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>)
        run_space<Kokkos::DefaultExecutionSpace>(Kokkos::DefaultExecutionSpace::name(), n, 11);
    }
  }
  Kokkos::finalize();
  return 0;
}
