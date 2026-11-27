// The supernodal route of laplace::sparse::sparse_ldlt against its simplicial route (itself checked
// against the dense inverse in test_selected_inverse.cpp): outcome, log |det|, pivots, number of factor
// entries, a solve, and the selected inverse on every entry of the simplicial pattern of L + L', on
// matrices whose trailing dense supernode ranges from one column to the whole matrix.
#include <doctest/doctest.h>

#include <laplace/sparse/sparse_ldlt.hpp>

#include "../../src/sparse/detail/dense_root_block.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace ls = laplace::sparse;

struct lower_matrix {
  int n = 0;
  std::map<std::pair<int, int>, double> e;  // (r, c) with r >= c
  void add(int r, int c, double v) {
    if (r < c) std::swap(r, c);
    e[{r, c}] += v;
  }
  void csc(std::vector<int>& cp, std::vector<int>& ri, std::vector<double>& vx) const {
    cp.assign(n + 1, 0), ri.clear(), vx.clear();
    for (int c = 0; c < n; ++c) {
      // the diagonal first in its column, as factorize() asks when the shift is not zero
      ri.push_back(c), vx.push_back(e.count({c, c}) ? e.at({c, c}) : 0.0);
      for (auto it = e.lower_bound({c + 1, c}); it != e.end(); ++it)
        if (it->first.second == c) ri.push_back(it->first.first), vx.push_back(it->second);
      cp[c + 1] = static_cast<int>(ri.size());
    }
  }
};

lower_matrix dominant(lower_matrix a, double extra) {
  std::vector<double> row(a.n, 0.0);
  for (const auto& [rc, v] : a.e)
    if (rc.first != rc.second) row[rc.first] += std::abs(v), row[rc.second] += std::abs(v);
  for (int j = 0; j < a.n; ++j) a.add(j, j, row[j] + extra);
  return a;
}

struct route_comparison {
  long entries = 0, outside = 0, nonzeros_simplicial = 0, nonzeros_supernodal = 0;
  double log_det = 0.0, log_det_error = 0.0, pivot_error = 0.0, solve_error = 0.0, max_x = 0.0;
  double z_error = 0.0, max_z = 0.0;
  double dz_error = 0.0, max_dz = 0.0, dot_value = 0.0, dot_error = 0.0;
};

route_comparison compare_routes(const lower_matrix& a, double shift, std::mt19937& rng) {
  std::vector<int> cp, ri;
  std::vector<double> vx;
  a.csc(cp, ri, vx);
  const int n = a.n;
  ls::sparse_ldlt simplicial, supernodal;
  simplicial.set_route(ls::sparse_ldlt_route::simplicial);
  supernodal.set_route(ls::sparse_ldlt_route::supernodal);
  REQUIRE(simplicial.factorize(n, cp.data(), ri.data(), vx.data(), shift) == ls::sparse_ldlt_outcome::positive_definite);
  REQUIRE(supernodal.factorize(n, cp.data(), ri.data(), vx.data(), shift) == ls::sparse_ldlt_outcome::positive_definite);
  CHECK(simplicial.route() == ls::sparse_ldlt_route::simplicial);
  CHECK(supernodal.route() == ls::sparse_ldlt_route::supernodal);
  route_comparison out;
  out.nonzeros_simplicial = simplicial.factor_nonzeros();
  out.nonzeros_supernodal = supernodal.factor_nonzeros();
  out.log_det = simplicial.log_abs_determinant();
  out.log_det_error = std::abs(supernodal.log_abs_determinant() - out.log_det);
  out.pivot_error = std::max(std::abs(supernodal.min_pivot() / simplicial.min_pivot() - 1.0),
                             std::abs(supernodal.max_pivot() / simplicial.max_pivot() - 1.0));
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::vector<double> x(n), y;
  for (double& v : x) v = unif(rng);
  y = x;
  REQUIRE(simplicial.solve(x.data()));
  REQUIRE(supernodal.solve(y.data()));
  for (int i = 0; i < n; ++i)
    out.max_x = std::max(out.max_x, std::abs(x[i])), out.solve_error = std::max(out.solve_error, std::abs(x[i] - y[i]));
  REQUIRE(simplicial.compute_selected_inverse());
  REQUIRE(supernodal.compute_selected_inverse());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double z = 0.0;
      try {
        z = simplicial.selected_inverse_entry(i, j);
      } catch (const std::out_of_range&) {
        continue;
      }
      ++out.entries;
      out.max_z = std::max(out.max_z, std::abs(z));
      try {
        out.z_error = std::max(out.z_error, std::abs(supernodal.selected_inverse_entry(i, j) - z));
      } catch (const std::out_of_range&) {
        ++out.outside;
      }
    }
  // the derivative along a random direction on the pattern of A, and its contraction with a random V
  std::vector<double> dvx(vx.size()), wvx(vx.size());
  for (std::size_t k = 0; k < vx.size(); ++k) dvx[k] = unif(rng), wvx[k] = unif(rng);
  REQUIRE(simplicial.compute_selected_inverse_derivative(dvx.data()));
  REQUIRE(supernodal.compute_selected_inverse_derivative(dvx.data()));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double dz = 0.0;
      try {
        dz = simplicial.selected_inverse_derivative_entry(i, j);
      } catch (const std::out_of_range&) {
        continue;
      }
      out.max_dz = std::max(out.max_dz, std::abs(dz));
      out.dz_error = std::max(out.dz_error, std::abs(supernodal.selected_inverse_derivative_entry(i, j) - dz));
    }
  out.dot_value = simplicial.selected_inverse_derivative_dot(wvx.data());
  out.dot_error = std::abs(supernodal.selected_inverse_derivative_dot(wvx.data()) - out.dot_value);
  return out;
}

void check_routes(const lower_matrix& a, double shift, std::mt19937& rng) {
  const auto r = compare_routes(a, shift, rng);
  MESSAGE("n ", a.n, ", factor entries ", r.nonzeros_simplicial, ", log det error ", r.log_det_error, " of ", r.log_det,
          ", pivots ", r.pivot_error, ", solve ", r.solve_error, " of ", r.max_x, ", Z ", r.z_error, " of ", r.max_z,
          " on ", r.entries, " entries; dZ ", r.dz_error, " of ", r.max_dz, ", contraction ", r.dot_error, " of ",
          r.dot_value);
  CHECK(r.nonzeros_supernodal == r.nonzeros_simplicial);
  CHECK(r.outside == 0);
  // bounds: about ten times the largest error measured over all the matrices on gcc (LAPACK and Eigen
  // kernels) and on the CUDA build with the device block (2026-11-27); the dense roots of 255 and 256
  // columns give the largest solve, Z and dZ errors
  CHECK(r.log_det_error <= 1e-14 * (1.0 + std::abs(r.log_det)));
  CHECK(r.pivot_error <= 2e-15);
  CHECK(r.solve_error <= 2e-14 * r.max_x);
  CHECK(r.z_error <= 3e-14 * r.max_z);
  CHECK(r.dz_error <= 7e-14 * r.max_dz);
  CHECK(r.dot_error <= 5e-14 * (1.0 + std::abs(r.dot_value)));
}

lower_matrix crossed(int G1, int G2, int cells, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::uniform_int_distribution<int> lev1(0, G1 - 1), lev2(0, G2 - 1);
  lower_matrix a;
  a.n = 2 + 2 * G1 + G2;  // two globals, two coordinates per level of the first grouping, one of the second
  for (int i = 0; i < cells; ++i) {
    const int g = lev1(rng), h = lev2(rng);
    const int u = 2 + 2 * g, v = u + 1, w = 2 + 2 * G1 + h;
    a.add(v, u, 0.3 * unif(rng)), a.add(w, u, 0.2 * unif(rng)), a.add(w, v, 0.2 * unif(rng));
    for (int b = 0; b < 2; ++b) a.add(u, b, unif(rng)), a.add(v, b, unif(rng)), a.add(w, b, unif(rng));
  }
  a.add(1, 0, unif(rng));
  return dominant(a, 0.2);
}

}  // namespace

// In a build with the device realization, the small matrices of these tests go to the device too: the
// threshold of 256 columns is lowered to 1 for the duration of a test case and restored afterwards.
struct device_on_small_blocks {
  int previous;
  device_on_small_blocks() : previous(laplace::sparse::detail::dense_root_block_set_device_min_width(1)) {}
  ~device_on_small_blocks() { laplace::sparse::detail::dense_root_block_set_device_min_width(previous); }
};

TEST_CASE("sparse_ldlt supernodal route against the simplicial route") {
  const device_on_small_blocks device_forced;
  std::mt19937 rng(20261127);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  SUBCASE("one coordinate") {
    lower_matrix a;
    a.n = 1;
    a.add(0, 0, 2.5);
    check_routes(a, 0.0, rng);
  }
  SUBCASE("tridiagonal") {
    lower_matrix a;
    a.n = 60;
    for (int j = 1; j < a.n; ++j) a.add(j, j - 1, unif(rng));
    check_routes(dominant(a, 0.3), 0.0, rng);
  }
  SUBCASE("dense: the whole matrix is the trailing supernode") {
    lower_matrix a;
    a.n = 40;
    for (int r = 1; r < a.n; ++r)
      for (int c = 0; c < r; ++c) a.add(r, c, unif(rng));
    check_routes(dominant(a, 0.5), 0.0, rng);
  }
  SUBCASE("two diagonal blocks: a forest of two trees") {
    lower_matrix a;
    a.n = 30;
    for (int r = 1; r < 15; ++r) a.add(r, r - 1, unif(rng));
    for (int r = 16; r < 30; ++r)
      for (int c = 15; c < r; ++c) a.add(r, c, unif(rng));
    check_routes(dominant(a, 0.4), 0.0, rng);
  }
  SUBCASE("random pattern with fill, and shifted") {
    lower_matrix a;
    a.n = 200;
    std::bernoulli_distribution keep(0.03);
    for (int r = 1; r < a.n; ++r)
      for (int c = 0; c < r; ++c)
        if (keep(rng)) a.add(r, c, unif(rng));
    a = dominant(a, 0.5);
    check_routes(a, 0.0, rng);
    check_routes(a, 0.75, rng);
  }
  SUBCASE("two crossed groupings, 150 x 150 levels, 1500 cells") { check_routes(crossed(150, 150, 1500, rng), 0.0, rng); }
  SUBCASE("two crossed groupings, 400 x 300 levels, 6000 cells") { check_routes(crossed(400, 300, 6000, rng), 0.0, rng); }
  SUBCASE("same pattern, new values, then the other route on the same object") {
    lower_matrix a = crossed(100, 80, 900, rng);
    std::vector<int> cp, ri;
    std::vector<double> vx;
    a.csc(cp, ri, vx);
    ls::sparse_ldlt f, g;
    f.set_route(ls::sparse_ldlt_route::supernodal);
    g.set_route(ls::sparse_ldlt_route::simplicial);
    REQUIRE(f.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::positive_definite);
    // new values on the same pattern: the entries below the diagonal shrunk by a factor in [0.9, 1], so the
    // matrix stays diagonally dominant
    for (int c = 0; c < a.n; ++c)
      for (int k = cp[c] + 1; k < cp[c + 1]; ++k) vx[k] *= 0.95 + 0.05 * unif(rng);
    REQUIRE(f.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::positive_definite);
    REQUIRE(g.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::positive_definite);
    CHECK(std::abs(f.log_abs_determinant() - g.log_abs_determinant()) <= 1e-12 * std::abs(g.log_abs_determinant()));
    f.set_route(ls::sparse_ldlt_route::simplicial);
    REQUIRE(f.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::positive_definite);
    CHECK(f.route() == ls::sparse_ldlt_route::simplicial);
    CHECK(f.log_abs_determinant() == g.log_abs_determinant());
  }
  SUBCASE("not positive definite: a negative pivot in the sparse columns, and in the dense block") {
    // coordinate 2 (a level of the first grouping) is eliminated among the sparse columns, coordinate 0 (a
    // global) belongs to the dense trailing supernode; a large negative diagonal makes either fail
    for (const int c : {2, 0}) {
      lower_matrix a = crossed(60, 50, 500, rng);
      std::vector<int> cp, ri;
      std::vector<double> vx;
      a.csc(cp, ri, vx);
      vx[cp[c]] = -1e3;
      ls::sparse_ldlt f, g;
      f.set_route(ls::sparse_ldlt_route::supernodal);
      g.set_route(ls::sparse_ldlt_route::simplicial);
      CHECK(f.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::not_positive_definite);
      CHECK(g.factorize(a.n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::not_positive_definite);
      // what a caller may still read after this outcome: the number of factor entries, the same on both
      // routes; the pivots and log |det| are documented as not available, NaN on the supernodal route
      CHECK(f.factor_nonzeros() == g.factor_nonzeros());
      CHECK(std::isnan(f.min_pivot()));
      CHECK(std::isnan(f.log_abs_determinant()));
      CHECK_FALSE(f.compute_selected_inverse());
      std::vector<double> b(a.n, 1.0);
      CHECK_FALSE(f.solve(b.data()));
    }
  }
}

TEST_CASE("sparse_ldlt supernodal route: the threshold of automatic, and the dense root block") {
  const device_on_small_blocks device_forced;
  std::mt19937 rng(20261128);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  namespace lsd = laplace::sparse::detail;
  const std::string realization = lsd::make_dense_root_block(8)->realization();
  MESSAGE("dense root block realization (device forced on small blocks): ", realization);

  SUBCASE("a dense matrix of 63 columns stays simplicial under automatic, one of 64 goes supernodal") {
    for (const int n : {63, 64}) {
      lower_matrix a;
      a.n = n;
      for (int r = 1; r < n; ++r)
        for (int c = 0; c < r; ++c) a.add(r, c, unif(rng));
      a = dominant(a, 0.5);
      std::vector<int> cp, ri;
      std::vector<double> vx;
      a.csc(cp, ri, vx);
      ls::sparse_ldlt f;
      REQUIRE(f.factorize(n, cp.data(), ri.data(), vx.data()) == ls::sparse_ldlt_outcome::positive_definite);
      CHECK(f.route() == (n >= 64 ? ls::sparse_ldlt_route::supernodal : ls::sparse_ldlt_route::simplicial));
      check_routes(a, 0.0, rng);  // both routes forced, root block of 63 and 64 columns
    }
  }
  SUBCASE("a root block that does not fit in the device memory goes to the host, and says so") {
    const long before = lsd::dense_root_block_device_fallbacks();
    lsd::dense_root_block_device_memory_cap(1);  // one byte: no block fits
    const std::string fallback = lsd::make_dense_root_block(8)->realization();
    check_routes(crossed(60, 50, 500, rng), 0.0, rng);
    lsd::dense_root_block_device_memory_cap(0);
    const long sent = lsd::dense_root_block_device_fallbacks() - before;
    MESSAGE("under the cap: ", fallback, "; blocks sent to the host ", sent);
    if (lsd::dense_root_block_device_available()) {
      CHECK(fallback != "cuda");
      CHECK(sent >= 2);
    } else {
      CHECK(fallback == realization);
      CHECK(sent == 0);
    }
  }
}

TEST_CASE("sparse_ldlt supernodal route: root blocks below and at the device threshold of 256 columns") {
  // the default threshold, not lowered: 255 columns stay on the host block, 256 go to the device in a build
  // that has it
  namespace lsd = laplace::sparse::detail;
  std::mt19937 rng(20261129);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  for (const int n : {255, 256}) {
    const std::string r = lsd::make_dense_root_block(n)->realization();
    MESSAGE("root of ", n, " columns: ", r);
    if (lsd::dense_root_block_device_available() && n >= 256) CHECK(r == "cuda");
    else CHECK(r != "cuda");
    lower_matrix a;
    a.n = n;
    for (int row = 1; row < n; ++row)
      for (int c = 0; c < row; ++c) a.add(row, c, unif(rng));
    check_routes(dominant(a, 0.5), 0.0, rng);
  }
}
