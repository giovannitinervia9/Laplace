// Shared by the bench and the test of Faa di Bruno's formula: the reader of
// tests/data/reference/faa_di_bruno.csv and the kernel that evaluates
// chain_rule_dN on every row.
#pragma once

#include "measure.hpp"

#include <laplace/math/faa_di_bruno.hpp>

namespace faa {

struct Rows {
  std::vector<std::array<double, 16>> fg;  // f[0..7] (f[0] = 0), then g[0..7]
  std::vector<std::array<double, 8>> hi, lo;  // d_n, n = 1..7 (index 0 unused)
};

inline Rows read(const std::string& path) {
  Rows r;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    char* p = line.data();
    char* end = nullptr;
    std::array<double, 16> fg{};
    std::array<double, 8> h{}, l{};
    for (int k = 1; k <= 7; ++k) fg[k] = std::strtod(p, &end), p = end + 1;
    for (int k = 1; k <= 7; ++k) fg[8 + k] = std::strtod(p, &end), p = end + 1;
    for (int n = 1; n <= 7; ++n) {
      h[n] = std::strtod(p, &end), p = end + 1;
      l[n] = std::strtod(p, &end), p = end + 1;
    }
    r.fg.push_back(fg), r.hi.push_back(h), r.lo.push_back(l);
  }
  return r;
}

KOKKOS_INLINE_FUNCTION double compose(int n, const double* f, const double* g) {
  switch (n) {
    case 1: return laplace::math::chain_rule_d1(f, g);
    case 2: return laplace::math::chain_rule_d2(f, g);
    case 3: return laplace::math::chain_rule_d3(f, g);
    case 4: return laplace::math::chain_rule_d4(f, g);
    case 5: return laplace::math::chain_rule_d5(f, g);
    case 6: return laplace::math::chain_rule_d6(f, g);
    default: return laplace::math::chain_rule_d7(f, g);
  }
}

// d_n for every row, in a kernel on the default execution space
inline std::vector<double> eval(const Rows& r, int n) {
  using V = Kokkos::View<double* [16], Kokkos::LayoutRight, bench::Exec::memory_space>;
  const std::size_t m = r.fg.size();
  V d("fg", m);
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < m; ++i)
    for (int c = 0; c < 16; ++c) h(i, c) = r.fg[i][c];
  Kokkos::deep_copy(d, h);
  Kokkos::View<double*, bench::Exec::memory_space> y("y", m);
  Kokkos::parallel_for(
      "faa", Kokkos::RangePolicy<bench::Exec>(0, m), KOKKOS_LAMBDA(const std::size_t i) {
        double f[8], g[8];
        for (int c = 0; c < 8; ++c) f[c] = d(i, c), g[c] = d(i, 8 + c);
        y(i) = compose(n, f, g);
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  return std::vector<double>(hy.data(), hy.data() + m);
}

// the error in ulps of d_n and in ulps of the sum of the absolute values of
// its terms, S = sum_k |f_k| B_(n,k)(|g_1|, ..., |g_(n-k+1)|)
struct Err {
  double ulp = 0, ulpS = 0;
  std::size_t worst = 0, bad = 0;
};
inline Err errors(const Rows& r, const std::vector<double>& y, int n) {
  Err e;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) {
      ++e.bad;
      continue;
    }
    double ga[8];
    for (int c = 0; c < 8; ++c) ga[c] = std::fabs(r.fg[i][8 + c]);
    double S = 0;
    for (int k = 1; k <= n; ++k) S += std::fabs(r.fg[i][k]) * laplace::math::bell_partial(n, k, ga);
    const double err = std::fabs((y[i] - r.hi[i][n]) - r.lo[i][n]);
    const double u = err / bench::ulp_of(std::max(std::fabs(r.hi[i][n]), 2.2250738585072014e-308));
    const double us = err / bench::ulp_of(S);
    if (u > e.ulp) e.ulp = u, e.worst = i;
    e.ulpS = std::max(e.ulpS, us);
  }
  return e;
}

}  // namespace faa
