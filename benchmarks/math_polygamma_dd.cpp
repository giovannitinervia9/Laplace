// Accuracy and speed of the double-double polygammas of orders 0..13
// (include/laplace/math/polygamma_dd.hpp) against the mpmath references of
// tools/reference/polygamma_dd.py (tests/data/reference/polygamma_dd.csv).
//
// Usage: math_polygamma_dd <reference dir> [points for the timing, default 2e5]
//
// Accuracy, on the default execution space: the worst error over the file of
// each function and order, single-order and all-orders form, in units of
// 2^-104 of the value (psi_0: of max(|f|, 1)); and, by band of x/h, the error of
// G_n written as the difference of two polygamma_dd values, which is what the
// difference function exists to avoid.
//
// Speed: ns per evaluation, x log-uniform in [1e-2, 1e8] and in [20, 1e8] (no
// recurrence there), h = 0.37, on one host thread (Serial) and on the default
// execution space; for the all-orders forms one evaluation is the whole call
// for N = 13. The plain double polygamma(6, x) is the reference cost.
#include <Kokkos_Core.hpp>
#include <laplace/math/polygamma.hpp>
#include <laplace/math/polygamma_dd.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "measure.hpp"

namespace {

namespace lm = laplace::math;
using lm::detail::ddm::dd;
using Exec = Kokkos::DefaultExecutionSpace;

struct Arg {
  int kind, n;
  double xh, xl, hh, hl;
};

// out(i, 0..1) single order, (2..3) all orders, (4..5) naive difference (kind 1)
void evaluate(const Kokkos::View<Arg*, Exec::memory_space>& arg, const Kokkos::View<double* [6], Exec::memory_space>& out) {
  Kokkos::parallel_for(
      "rows", Kokkos::RangePolicy<Exec>(0, arg.extent(0)), KOKKOS_LAMBDA(int i) {
        const Arg a = arg(i);
        const dd x(a.xh, a.xl), h(a.hh, a.hl);
        dd y, all[14], naive(0.0);
        if (a.kind == 0) {
          y = lm::polygamma_dd(a.n, x);
          lm::polygamma_orders_dd(13, x, all);
        } else if (a.kind == 1) {
          y = lm::polygamma_difference_dd(a.n, x, h);
          lm::polygamma_difference_orders_dd(13, x, h, all);
          naive = lm::polygamma_dd(a.n, x + h) - lm::polygamma_dd(a.n, x);
        } else {
          y = lm::polygamma_monomial_remainder_dd(a.n, x);
          lm::polygamma_monomial_remainder_orders_dd(13, x, all);
        }
        out(i, 0) = y.hi;
        out(i, 1) = y.lo;
        out(i, 2) = all[a.n].hi;
        out(i, 3) = all[a.n].lo;
        out(i, 4) = naive.hi;
        out(i, 5) = naive.lo;
      });
}

void accuracy(const std::string& dir) {
  std::vector<Arg> arg;
  std::vector<double> hi, mid, lo;
  std::ifstream in(dir + "/polygamma_dd.csv");
  if (!in) {
    std::fprintf(stderr, "cannot open %s/polygamma_dd.csv\n", dir.c_str());
    std::exit(2);
  }
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    for (char& c : line)
      if (c == ',') c = ' ';
    std::istringstream s(line);
    std::string t[9];
    for (auto& e : t) s >> e;
    auto num = [&](int i) { return std::strtod(t[i].c_str(), nullptr); };
    arg.push_back(Arg{std::atoi(t[0].c_str()), std::atoi(t[1].c_str()), num(2), num(3), num(4), num(5)});
    hi.push_back(num(6));
    mid.push_back(num(7));
    lo.push_back(num(8));
  }
  const int R = static_cast<int>(arg.size());
  Kokkos::View<Arg*, Exec::memory_space> darg("arg", R);
  Kokkos::View<double* [6], Exec::memory_space> out("out", R);
  auto ha = Kokkos::create_mirror_view(darg);
  for (int i = 0; i < R; ++i) ha(i) = arg[i];
  Kokkos::deep_copy(darg, ha);
  evaluate(darg, out);
  const auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);

  double worst[2][3][14] = {}, wx[3][14] = {}, wh[3][14] = {};
  // naive G: bands of log10(x/h) with edges below
  const double edge[8] = {-9, 1.5, 3.5, 5.5, 7.5, 9.5, 12.5, 17};
  double band_exact[7] = {}, band_naive[7] = {};
  int nonfinite = 0;
  for (int i = 0; i < R; ++i) {
    const Arg& a = arg[i];
    double scale = std::fabs(hi[i]);
    if (a.kind == 0 && a.n == 0 && scale < 1.0) scale = 1.0;
    auto err = [&](int c) { return std::fabs(((ho(i, c) - hi[i]) + (ho(i, c + 1) - mid[i])) - lo[i]) / (0x1p-104 * scale); };
    for (int f = 0; f < 2; ++f) {
      const double e = err(2 * f);
      if (!std::isfinite(e)) ++nonfinite;
      if (!(e <= worst[f][a.kind][a.n])) {
        worst[f][a.kind][a.n] = e;
        if (f == 0) {
          wx[a.kind][a.n] = a.xh;
          wh[a.kind][a.n] = a.hh;
        }
      }
    }
    if (a.kind == 1) {
      const double r = std::log10(a.xh / a.hh);
      for (int b = 0; b < 7; ++b)
        if (r >= edge[b] && r < edge[b + 1]) {
          band_exact[b] = std::max(band_exact[b], err(0));
          band_naive[b] = std::max(band_naive[b], err(4));
        }
    }
  }
  std::printf("accuracy on %s, %d rows, %d non-finite; units of 2^-104 (psi_0: of max(|f|,1))\n", Exec::name(), R,
              nonfinite);
  const char* name[3] = {"psi_n", "G_n", "r_n"};
  std::printf("%-22s", "n");
  for (int n = 0; n <= 13; ++n) std::printf(" %6d", n);
  std::printf("\n");
  for (int k = 0; k < 3; ++k)
    for (int f = 0; f < 2; ++f) {
      std::printf("%-6s %-15s", name[k], f == 0 ? "single order" : "all orders");
      for (int n = 0; n <= 13; ++n) std::printf(" %6.2f", worst[f][k][n]);
      std::printf("\n");
    }
  for (int k = 0; k < 3; ++k) {
    std::printf("worst %s (single): ", name[k]);
    for (int n = 0; n <= 13; ++n) std::printf(" [%d] x=%.4g h=%.3g", n, wx[k][n], wh[k][n]);
    std::printf("\n");
  }
  std::printf("G_n by log10(x/h): band, worst exact form, worst naive difference psi_n(x+h) - psi_n(x)\n");
  for (int b = 0; b < 7; ++b) std::printf("  [%5.1f, %5.1f)  %8.2f  %10.3g\n", edge[b], edge[b + 1], band_exact[b], band_naive[b]);
}

// the functors timed: x (and h) as doubles, the result's hi part
struct Psi {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double) const { return lm::polygamma_dd(n, dd(x)).hi; }
};
struct Diff {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double h) const { return lm::polygamma_difference_dd(n, dd(x), dd(h)).hi; }
};
struct Rem {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double) const { return lm::polygamma_monomial_remainder_dd(n, dd(x)).hi; }
};
struct PsiAll {
  KOKKOS_INLINE_FUNCTION double operator()(double x, double) const {
    dd o[14];
    lm::polygamma_orders_dd(13, dd(x), o);
    double s = 0.0;
    for (int n = 0; n <= 13; ++n) s += o[n].hi;
    return s;
  }
};
struct DiffAll {
  KOKKOS_INLINE_FUNCTION double operator()(double x, double h) const {
    dd o[14];
    lm::polygamma_difference_orders_dd(13, dd(x), dd(h), o);
    double s = 0.0;
    for (int n = 0; n <= 13; ++n) s += o[n].hi;
    return s;
  }
};
struct RemAll {
  KOKKOS_INLINE_FUNCTION double operator()(double x, double) const {
    dd o[14];
    lm::polygamma_monomial_remainder_orders_dd(13, dd(x), o);
    double s = 0.0;
    for (int n = 0; n <= 13; ++n) s += o[n].hi;
    return s;
  }
};
struct PsiDouble {
  KOKKOS_INLINE_FUNCTION double operator()(double x, double) const { return lm::own::polygamma(6, x); }
};

template <class Space>
void timing(const char* label, int M, double lo10, double hi10) {
  bench::HArgs hx("x", M);
  for (int i = 0; i < M; ++i) {
    hx(i, 0) = std::pow(10.0, lo10 + (hi10 - lo10) * (i + 0.5) / M);
    hx(i, 1) = 0.37;
    hx(i, 2) = 0.0;
  }
  Kokkos::View<double* [3], Kokkos::LayoutRight, typename Space::memory_space> x("x", M);
  Kokkos::deep_copy(x, hx);
  std::printf("%-8s x in [1e%g, 1e%g]:", label, lo10, hi10);
  for (int n : {0, 1, 6, 13}) {
    std::printf(" psi%d %.1f", n, bench::time_ns<Space>(x, Psi{n}, 3));
    std::printf(" G%d %.1f", n, bench::time_ns<Space>(x, Diff{n}, 3));
    std::printf(" r%d %.1f", n, bench::time_ns<Space>(x, Rem{n}, 3));
  }
  std::printf(" | all orders 0..13: psi %.1f G %.1f r %.1f | double polygamma(6) %.1f\n",
              bench::time_ns<Space>(x, PsiAll{}, 3), bench::time_ns<Space>(x, DiffAll{}, 3),
              bench::time_ns<Space>(x, RemAll{}, 3), bench::time_ns<Space>(x, PsiDouble{}, 3));
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_polygamma_dd <reference dir> [points]\n");
    return 2;
  }
  accuracy(argv[1]);
  const int M = argc > 2 ? std::atoi(argv[2]) : 200000;
  std::printf("ns per evaluation (h = 0.37)\n");
  timing<bench::HostOne>("1 thread", M / 10, -2.0, 8.0);
  timing<bench::HostOne>("1 thread", M / 10, std::log10(20.0), 8.0);
  timing<Exec>(Exec::name(), M, -2.0, 8.0);
  timing<Exec>(Exec::name(), M, std::log10(20.0), 8.0);
  return 0;
}
