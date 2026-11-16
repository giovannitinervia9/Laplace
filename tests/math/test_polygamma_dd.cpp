// The double-double polygammas of orders 0..13 (include/laplace/math/polygamma_dd.hpp)
// against the mpmath references of tools/reference/polygamma_dd.py, evaluated in
// a Kokkos kernel on the default execution space: psi_n, the difference
// G_n(x, h) = psi_n(x + h) - psi_n(x) and the remainder r_n = psi_n - mu_n, each
// by its single-order function and by its all-orders function.
//
// Errors are in units of 2^-104 of the value; psi_0 is measured in units of
// 2^-104 max(|f|, 1), since its relative error is unbounded at its root.
//
// Every kernel is in a named function with external linkage: an extended device
// lambda inside a doctest TEST_CASE shares its mangled tag with the same
// counter in another translation unit (CLAUDE.md, 2026-11-13).
#include <doctest/doctest.h>

#include <laplace/math/polygamma_dd.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace polygamma_dd_test {

namespace lm = laplace::math;
using lm::detail::ddm::dd;
using Exec = Kokkos::DefaultExecutionSpace;

struct Arg {
  int kind, n;  // kind 0 = psi_n, 1 = G_n, 2 = r_n
  double xh, xl, hh, hl;
};

struct Reference {
  std::vector<Arg> arg;
  std::vector<double> hi, mid, lo;
};

Reference read_reference() {
  Reference r;
  std::ifstream in(std::string(LAPLACE_REFERENCE_DIR) + "/polygamma_dd.csv");
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    for (char& c : line)
      if (c == ',') c = ' ';
    std::istringstream s(line);
    std::string t[9];
    for (auto& e : t) s >> e;
    auto num = [&](int i) { return std::strtod(t[i].c_str(), nullptr); };
    r.arg.push_back(Arg{std::atoi(t[0].c_str()), std::atoi(t[1].c_str()), num(2), num(3), num(4), num(5)});
    r.hi.push_back(num(6));
    r.mid.push_back(num(7));
    r.lo.push_back(num(8));
  }
  return r;
}

// out(i, 0..1): the single-order function; out(i, 2..3): the all-orders function
// with N = 13, entry n
void evaluate(const Kokkos::View<Arg*, Exec::memory_space>& arg, const Kokkos::View<double* [4], Exec::memory_space>& out) {
  Kokkos::parallel_for(
      "polygamma_dd_rows", Kokkos::RangePolicy<Exec>(0, arg.extent(0)), KOKKOS_LAMBDA(int i) {
        const Arg a = arg(i);
        const dd x(a.xh, a.xl), h(a.hh, a.hl);
        dd y, all[14];
        if (a.kind == 0) {
          y = lm::polygamma_dd(a.n, x);
          lm::polygamma_orders_dd(13, x, all);
        } else if (a.kind == 1) {
          y = lm::polygamma_difference_dd(a.n, x, h);
          lm::polygamma_difference_orders_dd(13, x, h, all);
        } else {
          y = lm::polygamma_monomial_remainder_dd(a.n, x);
          lm::polygamma_monomial_remainder_orders_dd(13, x, all);
        }
        out(i, 0) = y.hi;
        out(i, 1) = y.lo;
        out(i, 2) = all[a.n].hi;
        out(i, 3) = all[a.n].lo;
      });
}

struct Worst {
  double err[2][3][14] = {};  // [single, all-orders][kind][n]
  double at_x[2][3][14] = {}, at_h[2][3][14] = {};
  int nonfinite = 0;
};

Worst measure() {
  const Reference r = read_reference();
  const int R = static_cast<int>(r.arg.size());
  Kokkos::View<Arg*, Exec::memory_space> arg("arg", R);
  Kokkos::View<double* [4], Exec::memory_space> out("out", R);
  auto ha = Kokkos::create_mirror_view(arg);
  for (int i = 0; i < R; ++i) ha(i) = r.arg[i];
  Kokkos::deep_copy(arg, ha);
  evaluate(arg, out);
  const auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  Worst w;
  for (int i = 0; i < R; ++i) {
    const Arg& a = r.arg[i];
    double scale = std::fabs(r.hi[i]);
    if (a.kind == 0 && a.n == 0 && scale < 1.0) scale = 1.0;
    for (int f = 0; f < 2; ++f) {
      const double e = ((ho(i, 2 * f) - r.hi[i]) + (ho(i, 2 * f + 1) - r.mid[i])) - r.lo[i];
      const double u = std::fabs(e) / (0x1p-104 * scale);
      if (!std::isfinite(u)) ++w.nonfinite;
      if (!(u <= w.err[f][a.kind][a.n])) {
        w.err[f][a.kind][a.n] = u;
        w.at_x[f][a.kind][a.n] = a.xh;
        w.at_h[f][a.kind][a.n] = a.hh;
      }
    }
  }
  return w;
}

void check_kind(const Worst& w, int kind, const double* bound) {
  const char* name[3] = {"psi_n", "G_n", "r_n"};
  const char* form[2] = {"single order", "all orders"};
  for (int f = 0; f < 2; ++f)
    for (int n = 0; n <= 13; ++n) {
      INFO(name[kind], " (", form[f], "), n = ", n, ": ", w.err[f][kind][n], " units of 2^-104 at x = ",
           w.at_x[f][kind][n], ", h = ", w.at_h[f][kind][n]);
      CHECK(w.err[f][kind][n] <= bound[n]);
    }
}

const Worst& worst() {
  static const Worst w = measure();
  return w;
}

// NaN outside the domain, from a kernel
void domain(const Kokkos::View<double*, Exec::memory_space>& out) {
  Kokkos::parallel_for(
      "polygamma_dd_domain", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
        dd all[14];
        out(0) = lm::polygamma_dd(3, dd(0.0)).hi;
        out(1) = lm::polygamma_dd(3, dd(-1.0)).hi;
        out(2) = lm::polygamma_dd(14, dd(2.0)).hi;
        out(3) = lm::polygamma_dd(-1, dd(2.0)).hi;
        out(4) = lm::polygamma_difference_dd(3, dd(2.0), dd(0.0)).hi;
        out(5) = lm::polygamma_difference_dd(3, dd(2.0), dd(-0.5)).hi;
        out(6) = lm::polygamma_monomial_remainder_dd(3, dd(0.0)).hi;
        lm::polygamma_orders_dd(13, dd(-2.0), all);
        out(7) = all[13].hi;
        lm::polygamma_difference_orders_dd(13, dd(2.0), dd(0.0), all);
        out(8) = all[0].hi;
        lm::polygamma_monomial_remainder_orders_dd(13, dd(laplace::math::detail::quiet_nan()), all);
        out(9) = all[7].hi;
      });
}

}  // namespace polygamma_dd_test

using namespace polygamma_dd_test;

TEST_CASE("polygamma_dd: psi_n against mpmath, single order and all orders") {
  // Bounds: the worst of the two forms, measured on 2026-11-16 (identical on
  // gcc/OpenMP and CUDA), plus half: 0.96, 0.74, 1.04, 1.26, 1.75, 2.05, 2.15,
  // 2.10, 2.63, 3.08, 3.15, 3.53, 3.93, 4.47 units at n = 0..13.
  const double bound[14] = {1.5, 1.2, 1.6, 1.9, 2.7, 3.1, 3.3, 3.2, 4.0, 4.7, 4.8, 5.3, 5.9, 6.8};
  REQUIRE(worst().nonfinite == 0);
  check_kind(worst(), 0, bound);
}

TEST_CASE("polygamma_dd: G_n = psi_n(x + h) - psi_n(x) against mpmath, x/h up to 1e16") {
  // measured 0.79, 0.69, 1.09, 1.00, 1.75, 1.55, 1.47, 1.70, 2.13, 2.39, 2.85,
  // 2.84, 3.26, 3.29 (both forms, gcc and CUDA), plus half
  const double bound[14] = {1.2, 1.1, 1.7, 1.5, 2.7, 2.4, 2.3, 2.6, 3.2, 3.6, 4.3, 4.3, 4.9, 5.0};
  check_kind(worst(), 1, bound);
}

TEST_CASE("polygamma_dd: r_n = psi_n - mu_n against mpmath") {
  // measured 8.97, 3.44, 6.13, 5.00, 5.32, 9.05, 8.27, 7.66, 7.78, 7.29, 6.30,
  // 7.62, 8.49, 8.38 (worst of both forms, gcc and CUDA; worst at x = 12 to 15,
  // where the pole sum and the monomial difference are of one size), plus half
  const double bound[14] = {14, 5.2, 9.2, 7.5, 8.0, 14, 12.5, 11.5, 11.7, 11, 9.5, 11.5, 12.8, 12.6};
  check_kind(worst(), 2, bound);
}

TEST_CASE("polygamma_dd: NaN outside the domain") {
  Kokkos::View<double*, Exec::memory_space> out("out", 10);
  domain(out);
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < 10; ++i) {
    INFO("case ", i);
    CHECK(std::isnan(h(i)));
  }
}
