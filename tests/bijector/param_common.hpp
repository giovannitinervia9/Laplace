// Shared check of a scalar bijector with parameters against its mpmath
// references (tests/data/reference/bijector_<map>.csv, written by
// tools/reference/bijector_<map>.py through bijector_param_common.py),
// evaluated in a Kokkos kernel on the default execution space.
//
// Every row is a quantity (which: 0 forward, 1 inverse, 2 forward log-Jacobian,
// 3 inverse log-Jacobian, 4 and 5 the distances to the upper and lower bound)
// and a derivative d^k / (d arg^n0 d p1^n1 d p2^n2), k = n0 + n1 + n2 <= 6, at
// (x, p1, p2). Order 0 is the value, pure derivatives (n1 = n2 = 0) go through
// forward_dK / inverse_dK / ..._log_jacobian_dK, mixed ones through the
// _partials_dK functions; for the pure rows the component c[0] of the partials
// is checked as well. Error in ulpC: ulps of
//   |f| + |x df/dx| + |p1 df/dp1| + |p2 df/dp2|,
// the change that one ulp of relative perturbation of every input would cause
// (the neighbouring derivatives are read from the same file; the distances use
// the derivatives of h). Set LAPLACE_PRINT_ERRORS=1 to print the table.
#pragma once

#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/bijector/scalar.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace bijector_param_test {

namespace lb = laplace::bijector;
using map_t = lb::scalar_bijector_map;

// the bijector of a map with its parameters set directly (no normalization)
KOKKOS_INLINE_FUNCTION lb::scalar_bijector make(map_t m, double p1, double p2) {
  lb::scalar_bijector b{m};
  switch (m) {
    case map_t::affine: b.location = p1, b.scale = p2; break;
    case map_t::power:
    case map_t::box_cox:
    case map_t::yeo_johnson: b.exponent = p1; break;
    case map_t::softplus: b.sharpness = p1; break;
    case map_t::bounded: b.lower = p1, b.upper = p2; break;
    case map_t::lower_bounded: b.lower = p1; break;
    case map_t::upper_bounded: b.upper = p1; break;
    default: break;
  }
  return b;
}

template <int K>
KOKKOS_INLINE_FUNCTION double partial_of(const lb::scalar_bijector& b, int w, double x, int idx) {
  lb::bijector_partials<K> r;
  if constexpr (K == 1) {
    r = w == 0 ? b.forward_partials_d1(x) : w == 1 ? b.inverse_partials_d1(x) : w == 2 ? b.forward_log_jacobian_partials_d1(x) : b.inverse_log_jacobian_partials_d1(x);
  } else if constexpr (K == 2) {
    r = w == 0 ? b.forward_partials_d2(x) : w == 1 ? b.inverse_partials_d2(x) : w == 2 ? b.forward_log_jacobian_partials_d2(x) : b.inverse_log_jacobian_partials_d2(x);
  } else if constexpr (K == 3) {
    r = w == 0 ? b.forward_partials_d3(x) : w == 1 ? b.inverse_partials_d3(x) : w == 2 ? b.forward_log_jacobian_partials_d3(x) : b.inverse_log_jacobian_partials_d3(x);
  } else if constexpr (K == 4) {
    r = w == 0 ? b.forward_partials_d4(x) : w == 1 ? b.inverse_partials_d4(x) : w == 2 ? b.forward_log_jacobian_partials_d4(x) : b.inverse_log_jacobian_partials_d4(x);
  } else if constexpr (K == 5) {
    r = w == 0 ? b.forward_partials_d5(x) : w == 1 ? b.inverse_partials_d5(x) : w == 2 ? b.forward_log_jacobian_partials_d5(x) : b.inverse_log_jacobian_partials_d5(x);
  } else {
    r = w == 0 ? b.forward_partials_d6(x) : w == 1 ? b.inverse_partials_d6(x) : w == 2 ? b.forward_log_jacobian_partials_d6(x) : b.inverse_log_jacobian_partials_d6(x);
  }
  return r.c[idx];
}

KOKKOS_INLINE_FUNCTION double pure_of(const lb::scalar_bijector& b, int w, int k, double x) {
  switch (w * 8 + k) {
    case 0: return b.forward(x);
    case 1: return b.forward_d1(x);
    case 2: return b.forward_d2(x);
    case 3: return b.forward_d3(x);
    case 4: return b.forward_d4(x);
    case 5: return b.forward_d5(x);
    case 6: return b.forward_d6(x);
    case 8: return b.inverse(x);
    case 9: return b.inverse_d1(x);
    case 10: return b.inverse_d2(x);
    case 11: return b.inverse_d3(x);
    case 12: return b.inverse_d4(x);
    case 13: return b.inverse_d5(x);
    case 14: return b.inverse_d6(x);
    case 16: return b.forward_log_jacobian(x);
    case 17: return b.forward_log_jacobian_d1(x);
    case 18: return b.forward_log_jacobian_d2(x);
    case 19: return b.forward_log_jacobian_d3(x);
    case 20: return b.forward_log_jacobian_d4(x);
    case 21: return b.forward_log_jacobian_d5(x);
    case 22: return b.forward_log_jacobian_d6(x);
    case 24: return b.inverse_log_jacobian(x);
    case 25: return b.inverse_log_jacobian_d1(x);
    case 26: return b.inverse_log_jacobian_d2(x);
    case 27: return b.inverse_log_jacobian_d3(x);
    case 28: return b.inverse_log_jacobian_d4(x);
    case 29: return b.inverse_log_jacobian_d5(x);
    case 30: return b.inverse_log_jacobian_d6(x);
    case 32: return b.inverse_distance_upper(x);
    case 40: return b.inverse_distance_lower(x);
  }
  return laplace::math::detail::quiet_nan();
}

KOKKOS_INLINE_FUNCTION double partial_switch(const lb::scalar_bijector& b, int w, int k, double x, int idx) {
  switch (k) {
    case 1: return partial_of<1>(b, w, x, idx);
    case 2: return partial_of<2>(b, w, x, idx);
    case 3: return partial_of<3>(b, w, x, idx);
    case 4: return partial_of<4>(b, w, x, idx);
    case 5: return partial_of<5>(b, w, x, idx);
    default: return partial_of<6>(b, w, x, idx);
  }
}

using Rows = Kokkos::View<double* [7], Kokkos::LayoutRight, bench::Exec::memory_space>;
using Out = Kokkos::View<double* [2], Kokkos::LayoutRight, bench::Exec::memory_space>;

// y(i, 0): the value through the natural function; y(i, 1): for pure rows of
// order >= 1, the component c[0] of the partials (NaN-free 0 otherwise)
struct Eval {
  map_t m;
  Rows a;
  Out y;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const lb::scalar_bijector b = make(m, a(i, 1), a(i, 2));
    const double x = a(i, 0);
    const int w = static_cast<int>(a(i, 3)), n0 = static_cast<int>(a(i, 4)), n1 = static_cast<int>(a(i, 5)),
              n2 = static_cast<int>(a(i, 6)), k = n0 + n1 + n2;
    y(i, 1) = 0.0;
    if (w >= 4 || k == 0 || (n1 == 0 && n2 == 0)) {
      y(i, 0) = pure_of(b, w, k, x);
      if (w < 4 && k >= 1) y(i, 1) = partial_switch(b, w, k, x, 0);
    } else {
      const int idx = n2 * (k + 1) - n2 * (n2 - 1) / 2 + n1;  // bijector_partials<k>::index(n1, n2)
      y(i, 0) = partial_switch(b, w, k, x, idx);
    }
  }
};

struct Row {
  double a[7], hi, lo;
};

inline std::vector<Row> read(const std::string& path) {
  std::vector<Row> rows;
  std::ifstream in(path);
  REQUIRE_MESSAGE(static_cast<bool>(in), "cannot open ", path);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Row r;
    char* p = line.data();
    char* end = nullptr;
    for (double& v : r.a) v = std::strtod(p, &end), p = end + 1;
    r.hi = std::strtod(p, &end);
    r.lo = std::strtod(end + 1, &end);
    rows.push_back(r);
  }
  return rows;
}

inline const char* which_name(int w) {
  static const char* n[] = {"forward", "inverse", "fwd log-jac", "inv log-jac", "to upper", "to lower"};
  return n[w];
}

// bound[w][k]: the largest ulpC accepted for quantity w at total order k (< 0: no rows expected)
inline void check_map(const char* name, map_t m, const double (&bound)[6][7]) {
  const auto rows = read(std::string(LAPLACE_REFERENCE_DIR) + "/bijector_" + name + ".csv");
  REQUIRE(!rows.empty());
  using Key = std::tuple<int, int, int, int, double, double, double>;
  std::map<Key, double> ref;
  for (const auto& r : rows)
    ref[Key{int(r.a[3]), int(r.a[4]), int(r.a[5]), int(r.a[6]), r.a[0], r.a[1], r.a[2]}] = r.hi;
  std::vector<const Row*> tested;
  for (const auto& r : rows)
    if (r.a[4] + r.a[5] + r.a[6] <= 6) tested.push_back(&r);
  const std::size_t n = tested.size();
  Rows dv("rows", n);
  auto hv = Kokkos::create_mirror_view(dv);
  for (std::size_t i = 0; i < n; ++i)
    for (int j = 0; j < 7; ++j) hv(i, j) = tested[i]->a[j];
  Kokkos::deep_copy(dv, hv);
  Out dy("y", n);
  Kokkos::parallel_for("bijector_param_check", Kokkos::RangePolicy<bench::Exec>(0, n), Eval{m, dv, dy});
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);

  struct Acc {
    double ulpc = 0, ulp = 0, x = 0, p1 = 0, p2 = 0, c0 = 0;
    std::size_t rows = 0, bad = 0;
    int n1 = 0, n2 = 0;
  };
  std::map<int, Acc> acc;  // key w * 8 + k
  auto get = [&](int w, int n0, int n1, int n2, const Row& r) {
    const auto it = ref.find(Key{w, n0, n1, n2, r.a[0], r.a[1], r.a[2]});
    return it == ref.end() ? 0.0 : it->second;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const Row& r = *tested[i];
    const int w = int(r.a[3]), n0 = int(r.a[4]), n1 = int(r.a[5]), n2 = int(r.a[6]), k = n0 + n1 + n2;
    auto& e = acc[w * 8 + k];
    ++e.rows;
    const double y = hy(i, 0);
    if (!std::isfinite(y)) {
      ++e.bad;
      e.x = r.a[0], e.p1 = r.a[1], e.p2 = r.a[2];
      continue;
    }
    const int wb = w >= 4 ? 1 : w;  // the distances scale with the derivatives of h
    const int b0 = w >= 4 ? 0 : n0;
    const double scale = std::fabs(r.hi) + std::fabs(r.a[0] * get(wb, b0 + 1, n1, n2, r)) +
                         std::fabs(r.a[1] * get(wb, b0, n1 + 1, n2, r)) +
                         std::fabs(r.a[2] * get(wb, b0, n1, n2 + 1, r));
    const double d = std::fabs((y - r.hi) - r.lo);
    const double uc = d / bench::ulp_of(std::max(scale, 2.2250738585072014e-308));
    const double u = d / bench::ulp_of(std::max(std::fabs(r.hi), 2.2250738585072014e-308));
    if (uc > e.ulpc) e.ulpc = uc, e.x = r.a[0], e.p1 = r.a[1], e.p2 = r.a[2], e.n1 = n1, e.n2 = n2;
    e.ulp = std::max(e.ulp, u);
    if (w < 4 && k >= 1 && n1 == 0 && n2 == 0) {
      const double y1 = hy(i, 1);
      const double u1 = std::isfinite(y1) ? std::fabs((y1 - r.hi) - r.lo) / bench::ulp_of(std::max(scale, 2.2250738585072014e-308))
                                          : 1e300;
      e.c0 = std::max(e.c0, u1);
    }
  }
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (const auto& [key, e] : acc) {
    const int w = key / 8, k = key % 8;
    if (print)
      std::printf("%-14s %-11s k=%d  rows %5zu  max %9.3g ulpC (%9.3g ulp)  c[0] %8.3g  at x=%.6g p=(%.6g, %.6g) n1=%d n2=%d%s\n",
                  name, which_name(w), k, e.rows, e.ulpc, e.ulp, e.c0, e.x, e.p1, e.p2, e.n1, e.n2,
                  e.bad ? "  NOT FINITE" : "");
    INFO(name, " ", which_name(w), " order ", k, ": max ", e.ulpc, " ulpC at x = ", e.x, ", p = (", e.p1, ", ",
         e.p2, "), n1 = ", e.n1, ", n2 = ", e.n2, "; c[0] ", e.c0, "; ", e.bad, " not finite");
    CHECK(e.bad == 0);
    CHECK(bound[w][k] >= 0.0);
    CHECK(e.ulpc <= bound[w][k]);
    CHECK(e.c0 <= bound[w][k]);
  }
}

}  // namespace bijector_param_test
