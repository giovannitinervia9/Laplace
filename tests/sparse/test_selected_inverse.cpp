// The selected inverse of laplace::sparse::sparse_ldlt (Takahashi's recursion) against the dense inverse
// from laplace::dense (an independent Cholesky), on every entry of the pattern of L + L', the fill included;
// and its derivative along a random direction dA on the pattern of A against the dense -Z dA Z, with the
// contraction sum dZ_ij V_ij against -tr(Z dA Z V).
#include <doctest/doctest.h>

#include <laplace/dense/dense.hpp>
#include <laplace/sparse/sparse_ldlt.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace ls = laplace::sparse;

// a symmetric matrix by its lower triangle, entry (r, c) with r >= c
struct lower_entries {
  int n = 0;
  std::map<std::pair<int, int>, double> e;
  void add(int r, int c, double v) {
    if (r < c) std::swap(r, c);
    e[{r, c}] += v;
  }
};

// diagonal dominance makes it positive definite
lower_entries dominant(lower_entries a, double extra) {
  std::vector<double> row(a.n, 0.0);
  for (const auto& [rc, v] : a.e)
    if (rc.first != rc.second) row[rc.first] += std::abs(v), row[rc.second] += std::abs(v);
  for (int j = 0; j < a.n; ++j) a.add(j, j, row[j] + extra);
  return a;
}

struct check_result {
  long pattern_entries = 0, stored_but_outside = 0;
  double max_error = 0.0, max_z = 0.0;
  double max_derivative_error = 0.0, max_dz = 0.0, dot_error = 0.0, dot_value = 0.0;
};

check_result compare(const lower_entries& a, double shift) {
  const int n = a.n;
  std::vector<int> cp(n + 1, 0), ri;
  std::vector<double> vx;
  for (int c = 0; c < n; ++c) {
    // the diagonal first in its column, as factorize() asks when shift != 0
    ri.push_back(c), vx.push_back(a.e.count({c, c}) ? a.e.at({c, c}) : 0.0);
    for (const auto& [rc, v] : a.e)
      if (rc.second == c && rc.first != c) ri.push_back(rc.first), vx.push_back(v);
    cp[c + 1] = static_cast<int>(ri.size());
  }
  ls::sparse_ldlt f;
  CHECK_THROWS_AS((void)f.selected_inverse_entry(0, 0), std::out_of_range);
  CHECK_THROWS_AS((void)f.selected_inverse_derivative_entry(0, 0), std::out_of_range);
  REQUIRE(f.factorize(n, cp.data(), ri.data(), vx.data(), shift) == ls::sparse_ldlt_outcome::positive_definite);
  CHECK_THROWS_AS((void)f.selected_inverse_entry(0, 0), std::out_of_range);
  REQUIRE(f.compute_selected_inverse());

  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", n, n), L("L", n, n), B("B", n, n);
  for (const auto& [rc, v] : a.e) A(rc.first, rc.second) = A(rc.second, rc.first) = v;
  for (int j = 0; j < n; ++j) A(j, j) += shift, B(j, j) = 1.0;
  REQUIRE(laplace::dense::spd_solve(A, L, B).success);

  check_result out;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) out.max_z = std::max(out.max_z, std::abs(B(i, j)));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double z = 0.0;
      try {
        z = f.selected_inverse_entry(i, j);
      } catch (const std::out_of_range&) {
        // outside the pattern: then (i, j) is no entry of A
        out.stored_but_outside += static_cast<long>(a.e.count({std::max(i, j), std::min(i, j)}));
        continue;
      }
      ++out.pattern_entries;
      out.max_error = std::max(out.max_error, std::abs(z - B(i, j)));
    }

  // the derivative along a random symmetric dA, and a random symmetric V, both on the pattern of A
  std::mt19937 rng(7 + n);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::vector<double> dvx(vx.size()), wvx(vx.size());
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> dA("dA", n, n), V("V", n, n);
  for (int c = 0; c < n; ++c)
    for (int k = cp[c]; k < cp[c + 1]; ++k) {
      dvx[k] = unif(rng), wvx[k] = unif(rng);
      dA(ri[k], c) = dA(c, ri[k]) = dvx[k];
      V(ri[k], c) = V(c, ri[k]) = wvx[k];
    }
  REQUIRE(f.compute_selected_inverse_derivative(dvx.data()));
  // dense dZ = -B dA B, and -tr(B dA B V) = sum dZ_ij V_ij
  std::vector<double> T(static_cast<std::size_t>(n) * n, 0.0), dZ(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      if (dA(i, k) == 0.0) continue;
      for (int j = 0; j < n; ++j) T[i + n * j] += dA(i, k) * B(k, j);  // T = dA B
    }
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j) dZ[i + n * j] -= B(i, k) * T[k + n * j];
  double dot = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) out.max_dz = std::max(out.max_dz, std::abs(dZ[i + n * j])), dot += dZ[i + n * j] * V(i, j);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double z = 0.0;
      try {
        z = f.selected_inverse_derivative_entry(i, j);
      } catch (const std::out_of_range&) {
        continue;
      }
      out.max_derivative_error = std::max(out.max_derivative_error, std::abs(z - dZ[i + n * j]));
    }
  out.dot_value = dot;
  out.dot_error = std::abs(f.selected_inverse_derivative_dot(wvx.data()) - dot);
  return out;
}

void check(const lower_entries& a, double shift, long min_entries) {
  const auto r = compare(a, shift);
  MESSAGE("n ", a.n, ", pattern entries ", r.pattern_entries, ", max error ", r.max_error, " of max |Z| ", r.max_z, " (relative ", r.max_error / r.max_z, ")");
  CHECK(r.stored_but_outside == 0);
  CHECK(r.pattern_entries >= min_entries);
  CHECK(r.max_error <= 1e-13 * r.max_z);
  MESSAGE("derivative: max error ", r.max_derivative_error, " of max |dZ| ", r.max_dz, " (relative ",
          r.max_dz > 0.0 ? r.max_derivative_error / r.max_dz : 0.0, "); contraction ", r.dot_value, ", error ",
          r.dot_error);
  CHECK(r.max_derivative_error <= 1e-12 * r.max_dz);
  CHECK(r.dot_error <= 1e-11 * (1.0 + std::abs(r.dot_value)));
}

}  // namespace

TEST_CASE("sparse_ldlt selected inverse: tridiagonal, arrowhead, random, crossed, shifted, against the dense inverse") {
  std::mt19937 rng(20261126);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  SUBCASE("one coordinate") {
    lower_entries a;
    a.n = 1;
    a.add(0, 0, 2.5);
    check(a, 0.0, 1);
  }
  SUBCASE("tridiagonal, no fill; an entry outside the pattern throws") {
    lower_entries a;
    a.n = 50;
    for (int j = 1; j < a.n; ++j) a.add(j, j - 1, unif(rng));
    a = dominant(a, 0.3);
    check(a, 0.0, a.n + 2 * (a.n - 1));
    std::vector<int> cp(a.n + 1, 0), ri;
    std::vector<double> vx;
    for (int c = 0; c < a.n; ++c) {
      for (const auto& [rc, v] : a.e)
        if (rc.second == c) ri.push_back(rc.first), vx.push_back(v);
      cp[c + 1] = static_cast<int>(ri.size());
    }
    ls::sparse_ldlt f;
    REQUIRE(f.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::positive_definite);
    REQUIRE(f.compute_selected_inverse());
    CHECK_THROWS_AS((void)f.selected_inverse_entry(0, 10), std::out_of_range);
  }
  SUBCASE("arrowhead: 100 groups of 2 and a border of 4") {
    lower_entries a;
    a.n = 4 + 200;
    for (int g = 0; g < 100; ++g) {
      const int u = 4 + 2 * g;
      a.add(u + 1, u, unif(rng));
      for (int b = 0; b < 4; ++b) a.add(u, b, unif(rng)), a.add(u + 1, b, unif(rng));
    }
    for (int b = 1; b < 4; ++b)
      for (int c = 0; c < b; ++c) a.add(b, c, unif(rng));
    a = dominant(a, 1.0);
    check(a, 0.0, static_cast<long>(a.e.size()));
  }
  SUBCASE("random pattern, with fill") {
    lower_entries a;
    a.n = 150;
    std::bernoulli_distribution keep(0.04);
    for (int r = 1; r < a.n; ++r)
      for (int c = 0; c < r; ++c)
        if (keep(rng)) a.add(r, c, unif(rng));
    a = dominant(a, 0.5);
    check(a, 0.0, static_cast<long>(a.e.size()));
    SUBCASE("and with a shift") { check(a, 0.75, static_cast<long>(a.e.size())); }
  }
  SUBCASE("two crossed groupings: 40 x 40 levels, 400 cells") {
    lower_entries a;
    const int G = 40;
    a.n = 2 + 2 * G;
    std::uniform_int_distribution<int> lev(0, G - 1);
    for (int i = 0; i < 400; ++i) {
      const int u = 2 + lev(rng), v = 2 + G + lev(rng);
      a.add(v, u, 0.2 * unif(rng));
      a.add(u, 0, unif(rng)), a.add(v, 0, unif(rng)), a.add(u, 1, unif(rng)), a.add(v, 1, unif(rng));
    }
    a.add(1, 0, unif(rng));
    a = dominant(a, 0.1);
    check(a, 0.0, static_cast<long>(a.e.size()));
  }
}
