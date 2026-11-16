// Cyclic B-spline bases (include/laplace/basis/cyclic_bspline.hpp) against the
// exact rational references of tools/reference/basis_cyclic_bspline.py, which
// evaluate the periodic functions on the bi-infinite knot sequence without
// reducing x and without folding. Rows of order 0..min(d, 6), integral rows
// and integral_column of every column are evaluated in a Kokkos kernel on the
// default execution space; totals and Gram matrices on the host. 47 cases,
// degrees 0, 1, 2, 3, 5, 10: uniform on [0, 1), [1000, 1001), [0, 365.25),
// K = d + 1; random knots on [0, 365); multiplicity 2 and d + 1 inside the
// period and at lower. Points: 20 random in [lower - 2 omega, lower + 3 omega],
// the knots of one period and one ulp on either side, upper.
// Errors: rows in ulps of the row's largest value; integrals in ulps of
// max(largest column total, largest reference value at the point); Gram
// entries in ulps of the largest entry.
// Then: the wrap (derivatives 0..d - mu continuous at lower), Gram matrix
// positive definite (with the wrap, the span is the periodic spline space),
// expansions, the arithmetic interval, operator_penalty_matrix, constructors.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/basis/cyclic_bspline.hpp>
#include <laplace/basis/operator.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace basis_cyclic_bspline_test {

namespace lb = laplace::basis;
constexpr int W = lb::bspline_max_degree + 1;  // widest row
constexpr int KM = 24;                         // largest K of the cases

struct case_data {
  std::string name;
  int degree = 0, dim = 0, kind = 0;
  double lower = 0, period = 1;
  std::vector<double> knots;
  std::vector<double> x;
  std::map<double, int> point;
  std::vector<std::vector<double>> row;       // [point][k * K + column], dense
  std::vector<std::vector<double>> integral;  // [point][column]
  std::vector<double> total;
  std::map<std::pair<int, int>, std::vector<double>> gram;  // row-major
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> f;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ',')) f.push_back(x);
  return f;
}
double hex(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

std::vector<case_data> read_cases() {
  std::vector<case_data> cs;
  std::ifstream cf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_cyclic_bspline_cases.csv");
  REQUIRE(cf.good());
  std::string line;
  std::getline(cf, line);
  while (std::getline(cf, line)) {
    auto f = split(line);
    const int c = std::stoi(f[0]);
    if (c >= static_cast<int>(cs.size())) {
      cs.emplace_back();
      cs.back().name = f[1];
      cs.back().degree = std::stoi(f[2]);
      cs.back().dim = std::stoi(f[3]);
      cs.back().kind = std::stoi(f[4]);
      cs.back().lower = hex(f[5]);
      cs.back().period = hex(f[6]);
    }
    cs[c].knots.push_back(hex(f[8]));
  }
  std::ifstream rf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_cyclic_bspline.csv");
  REQUIRE(rf.good());
  std::getline(rf, line);
  while (std::getline(rf, line)) {
    auto f = split(line);
    auto& c = cs[std::stoi(f[0])];
    const int q = std::stoi(f[1]), k = std::stoi(f[2]), l = std::stoi(f[3]);
    const double x = hex(f[4]), v = hex(f[7]);
    const int first = std::stoi(f[5]), i = std::stoi(f[6]);
    const int K = c.dim;
    if (q == 0 || q == 1) {
      auto it = c.point.find(x);
      int m;
      if (it == c.point.end()) {
        m = static_cast<int>(c.x.size());
        c.point[x] = m;
        c.x.push_back(x);
        c.row.emplace_back(7 * K, 0.0);
        c.integral.emplace_back(K, 0.0);
      } else {
        m = it->second;
      }
      if (q == 0) c.row[m][k * K + (first + i) % K] += v;
      else c.integral[m][i] = v;
    } else if (q == 2) {
      c.total.push_back(v);
    } else {
      auto& g = c.gram[{k, l}];
      if (g.empty()) g.assign(static_cast<size_t>(K) * K, 0.0);
      g[static_cast<size_t>(first) * K + i] = v;
    }
  }
  return cs;
}

lb::cyclic_bspline_basis_owner make_owner(const case_data& c) {
  if (c.kind == 0) return lb::cyclic_bspline_basis_owner::uniform(c.lower, c.period, c.dim, c.degree);
  return lb::cyclic_bspline_basis_owner::from_knots(c.knots, c.period, c.degree);
}

using DV = Kokkos::View<double*>;
using DV2 = Kokkos::View<double**, Kokkos::LayoutRight>;
using IV2 = Kokkos::View<int**, Kokkos::LayoutRight>;

// per point: rows of order 0..6 and the integral row (first and d + 1 values
// in out(m, k * W ..)), then integral_column of every column in out(m, 8 * W ..)
struct eval_functor {
  lb::cyclic_bspline_basis b;
  DV x;
  DV2 out;
  IV2 first;
  KOKKOS_INLINE_FUNCTION void operator()(int m) const {
    double r[W];
    for (int k = 0; k < 8; ++k) {
      first(m, k) = k < 7 ? lb::basis_row_of_order(b, k, x(m), r) : b.integral_row(x(m), r);
      for (int q = 0; q < W; ++q) out(m, k * W + q) = q < b.row_width() ? r[q] : 0.0;
    }
    for (int j = 0; j < b.dimension(); ++j) out(m, 8 * W + j) = b.integral_column(j, x(m));
  }
};

void eval_device(const lb::cyclic_bspline_basis& b, const std::vector<double>& xs, std::vector<std::vector<double>>& out,
                 std::vector<std::vector<int>>& first) {
  const int n = static_cast<int>(xs.size());
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int m = 0; m < n; ++m) hx(m) = xs[m];
  Kokkos::deep_copy(x, hx);
  DV2 o("out", n, 8 * W + KM);
  IV2 f("first", n, 8);
  Kokkos::parallel_for("cyclic_bspline_eval", n, eval_functor{b, x, o, f});
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  out.assign(n, std::vector<double>(8 * W + KM));
  first.assign(n, std::vector<int>(8));
  for (int m = 0; m < n; ++m) {
    for (int j = 0; j < 8 * W + KM; ++j) out[m][j] = ho(m, j);
    for (int k = 0; k < 8; ++k) first[m][k] = hf(m, k);
  }
}

double ulp_of(double s) { return std::nextafter(s, INFINITY) - s; }
// the larger of two errors; a NaN error wins (std::max would drop it)
double worse(double e, double err) { return (err > e || err != err) ? err : e; }
bool print_errors() { return std::getenv("LAPLACE_PRINT_ERRORS") != nullptr; }

// bounds in ulps: 1.5 times the measured maximum over gcc and CUDA
// (2026-11-14). Measured, d <= 5 and d = 10: rows 8 and 11; integrals 6 and 6
// (totals 3); Gram matrices 8 and 9.
double row_bound(int d) { return d <= 5 ? 12.0 : 17.0; }
double integral_bound(int) { return 9.0; }
double gram_bound(int d) { return d <= 5 ? 12.0 : 14.0; }

// lower triangle of a symmetric matrix (column-major), Cholesky; false when
// a pivot is not positive
bool positive_definite(std::vector<double> a, int n) {
  for (int j = 0; j < n; ++j) {
    double s = a[j + static_cast<size_t>(j) * n];
    for (int k = 0; k < j; ++k) s -= a[j + static_cast<size_t>(k) * n] * a[j + static_cast<size_t>(k) * n];
    if (!(s > 0)) return false;
    const double p = std::sqrt(s);
    a[j + static_cast<size_t>(j) * n] = p;
    for (int i = j + 1; i < n; ++i) {
      double t = a[i + static_cast<size_t>(j) * n];
      for (int k = 0; k < j; ++k) t -= a[i + static_cast<size_t>(k) * n] * a[j + static_cast<size_t>(k) * n];
      a[i + static_cast<size_t>(j) * n] = t / p;
    }
  }
  return true;
}

void check_case(const case_data& c) {
  INFO("case " << c.name << " d=" << c.degree);
  const int d = c.degree, K = c.dim;
  REQUIRE(K <= KM);
  auto owner = make_owner(c);
  REQUIRE(owner.knots().size() == c.knots.size());
  for (size_t j = 0; j < c.knots.size(); ++j) CHECK(owner.knots()[j] == c.knots[j]);

  std::vector<std::vector<double>> out;
  std::vector<std::vector<int>> first;
  eval_device(owner.device(), c.x, out, first);

  double tmax = 0;
  for (double v : c.total) tmax = std::max(tmax, std::fabs(v));
  // errors at the points of one period (exact reduction) and at the others
  double e_row[2] = {0, 0}, e_int[2] = {0, 0};
  std::vector<double> dense(K), dense_int(K);
  for (size_t m = 0; m < c.x.size(); ++m) {
    const double x = c.x[m];
    const int far = (x < c.lower || x > c.lower + c.period) ? 1 : 0;
    for (int k = 0; k <= 6; ++k) {
      std::fill(dense.begin(), dense.end(), 0.0);
      for (int q = 0; q <= d; ++q) dense[(first[m][k] + q) % K] += out[m][k * W + q];
      if (k > d) {  // above the degree: exact zeros
        for (int q = 0; q <= d; ++q) CHECK(out[m][k * W + q] == 0.0);
        continue;
      }
      double scale = 0;
      for (int j = 0; j < K; ++j) scale = std::max(scale, std::fabs(c.row[m][k * K + j]));
      if (scale == 0) scale = 1;
      for (int j = 0; j < K; ++j)
        e_row[far] = worse(e_row[far], std::fabs(dense[j] - c.row[m][k * K + j]) / ulp_of(scale));
    }
    double iscale = tmax;
    for (int j = 0; j < K; ++j) iscale = std::max(iscale, std::fabs(c.integral[m][j]));
    for (int j = 0; j < K; ++j) {
      e_int[far] = worse(e_int[far], std::fabs(out[m][8 * W + j] - c.integral[m][j]) / ulp_of(iscale));
    }
    // the integral row holds integral_column of its window, bit for bit
    for (int q = 0; q <= d; ++q) CHECK(out[m][7 * W + q] == out[m][8 * W + (first[m][7] + q) % K]);
    CHECK(first[m][7] == first[m][0]);
  }

  const auto hb = owner.host();
  double e_tot = 0;
  for (int j = 0; j < K; ++j) e_tot = worse(e_tot, std::fabs(hb.integral_column_total(j) - c.total[j]) / ulp_of(tmax));
  double e_gram = 0;
  std::vector<double> g(static_cast<size_t>(K) * K);
  for (const auto& [kl, ref] : c.gram) {
    hb.cross_gram(kl.first, kl.second, g.data());
    double gm = 0;
    for (double v : ref) gm = std::max(gm, std::fabs(v));
    if (gm == 0) gm = 1;
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b)
        e_gram = worse(e_gram, std::fabs(g[a + static_cast<size_t>(b) * K] - ref[static_cast<size_t>(a) * K + b]) / ulp_of(gm));
  }
  // the Gram matrix of the values is positive definite: the K functions are
  // independent
  hb.cross_gram(0, 0, g.data());
  CHECK(positive_definite(g, K));

  if (print_errors())
    std::printf("d=%2d K=%2d %-18s rows %6.1f far %8.1f  integral %6.1f far %8.1f  total %5.1f  gram %5.1f\n", d, K,
                c.name.c_str(), e_row[0], e_row[1], e_int[0], e_int[1], e_tot, e_gram);
  CHECK(e_row[0] <= row_bound(d));
  CHECK(e_row[1] <= row_bound(d));
  CHECK(e_int[0] <= integral_bound(d));
  CHECK(e_int[1] <= integral_bound(d));
  CHECK(e_tot <= integral_bound(d));
  CHECK(e_gram <= gram_bound(d));

  // the wrap: at lower, derivatives 0..d - mu equal ext's left limits at upper
  // (ext's last interval), mu the multiplicity of lower
  int mu = 0;
  for (double t : c.knots) mu += t == c.knots[0];
  double rl[W], ru[W];
  for (int k = 0; k <= std::min(d - mu, 6); ++k) {
    std::fill(dense.begin(), dense.end(), 0.0);
    std::fill(dense_int.begin(), dense_int.end(), 0.0);
    const int fl = lb::basis_row_of_order(hb, k, c.lower, rl);
    const int fu = hb.cyclic_column(lb::basis_row_of_order(hb.ext, k, hb.upper, ru));
    double scale = 0;
    for (int q = 0; q <= d; ++q) {
      dense[(fl + q) % K] += rl[q];
      dense_int[(fu + q) % K] += ru[q];
      scale = std::max(scale, std::fabs(rl[q]));
    }
    double e = 0;
    for (int j = 0; j < K; ++j) e = std::max(e, std::fabs(dense[j] - dense_int[j]));
    INFO("wrap k=" << k);
    CHECK(e <= 64 * ulp_of(scale));
  }

  // expansions on the host against the dense rows
  std::vector<double> beta(K);
  for (int j = 0; j < K; ++j) beta[j] = std::sin(1.0 + 0.7 * j);
  for (size_t m = 0; m < c.x.size(); m += 3) {
    const double x = c.x[m];
    double r[W];
    for (int k = 0; k <= 6; ++k) {
      const int f = lb::basis_row_of_order(hb, k, x, r);
      double s = 0, a = 0;
      for (int q = 0; q <= d; ++q) {
        s += beta[(f + q) % K] * r[q];
        a += std::fabs(beta[(f + q) % K] * r[q]);
      }
      double e;
      switch (k) {
        case 0: e = hb.expansion(x, beta.data()); break;
        case 1: e = hb.expansion_d1(x, beta.data()); break;
        case 2: e = hb.expansion_d2(x, beta.data()); break;
        case 3: e = hb.expansion_d3(x, beta.data()); break;
        case 4: e = hb.expansion_d4(x, beta.data()); break;
        case 5: e = hb.expansion_d5(x, beta.data()); break;
        default: e = hb.expansion_d6(x, beta.data()); break;
      }
      CHECK(std::fabs(e - s) <= 4 * ulp_of(a + 1e-300));
    }
    double s = 0, a = 0;
    for (int j = 0; j < K; ++j) {
      const double v = hb.integral_column(j, x);
      s += beta[j] * v;
      a += std::fabs(beta[j] * v);
    }
    CHECK(std::fabs(hb.integral_expansion(x, beta.data()) - s) <= 4 * K * ulp_of(a + 1e-300));
  }

  // the arithmetic interval gives exactly the rows of the binary search
  if (hb.ext.step > 0.0) {
    auto bs = hb;
    bs.ext.step = 0.0;
    double r1[W], r2[W];
    int mism = 0;
    for (double x : c.x)
      for (int k = 0; k <= 6; ++k) {
        const int f1 = lb::basis_row_of_order(hb, k, x, r1), f2 = lb::basis_row_of_order(bs, k, x, r2);
        if (f1 != f2) ++mism;
        for (int q = 0; q <= d; ++q)
          if (r1[q] != r2[q]) ++mism;
      }
    CHECK(mism == 0);
  }
}

TEST_CASE("cyclic_bspline: rows, integrals, totals and Gram matrices against exact references") {
  auto cs = read_cases();
  REQUIRE(cs.size() == 47);
  for (const auto& c : cs) check_case(c);
}

TEST_CASE("cyclic_bspline: far points, non-finite x, operator_penalty_matrix, constructors") {
  using owner = lb::cyclic_bspline_basis_owner;
  const auto o = owner::uniform(0.0, 365.25, 12, 3);
  const auto b = o.host();
  CHECK(b.dimension() == 12);
  CHECK(b.row_width() == 4);
  CHECK(b.gram_bandwidth() == 11);

  // one period more adds the column totals to the integrals
  std::vector<double> beta(12);
  double tot = 0;
  for (int j = 0; j < 12; ++j) {
    beta[j] = std::cos(0.3 * j);
    tot += beta[j] * b.integral_column_total(j);
  }
  for (double x : {17.0, 200.5, -40.0, 3000.0}) {
    const double i0 = b.integral_expansion(x, beta.data()), i1 = b.integral_expansion(x + 365.25, beta.data());
    CHECK(std::fabs(i1 - i0 - tot) <= 1e-12 * (std::fabs(i0) + std::fabs(i1) + std::fabs(tot)));
  }

  // non-finite x
  double r[4];
  for (double x : {std::nan(""), double(INFINITY), -double(INFINITY)}) {
    b.row(x, r);
    for (double v : r) CHECK(std::isnan(v));
    CHECK(std::isnan(b.expansion(x, beta.data())));
    CHECK(std::isnan(b.integral_expansion(x, beta.data())));
  }

  // operator_penalty_matrix of D^2 is G_22; above the degree it throws
  std::vector<double> s(144), g(144);
  lb::operator_penalty_matrix(b, lb::derivative_operator(2), s.data());
  b.cross_gram(2, 2, g.data());
  for (int i = 0; i < 144; ++i) CHECK(s[i] == g[i]);
  CHECK_THROWS_AS(lb::operator_penalty_matrix(b, lb::derivative_operator(4), s.data()), std::invalid_argument);

  // regularity points: the knots, lower included, with C^(d - mu)
  const auto po = owner::from_knots({0.0, 0.0, 0.3, 0.6}, 1.0, 3);
  const auto p = po.host().regularity_points();
  REQUIRE(p.size() == 3);
  CHECK(p[0].x == 0.0);
  CHECK(p[0].continuity == 1);
  CHECK(p[1].continuity == 2);

  // Greville abscissae lie in [lower, upper)
  std::vector<double> xi(12);
  b.greville(xi.data());
  for (double v : xi) CHECK((v >= 0.0 && v < 365.25));

  CHECK_THROWS_AS(owner::uniform(0.0, 1.0, 3, 3), std::invalid_argument);
  CHECK_THROWS_AS(owner::uniform(0.0, 0.0, 8, 3), std::invalid_argument);
  CHECK_THROWS_AS(owner::uniform(0.0, 1.0, 8, 11), std::invalid_argument);
  CHECK_THROWS_AS(owner::from_knots({0.0, 0.5, 1.0, 0.7}, 1.0, 2), std::invalid_argument);
  CHECK_THROWS_AS(owner::from_knots({0.0, 0.2, 0.5, 1.0}, 1.0, 2), std::invalid_argument);
  CHECK_THROWS_AS(owner::from_knots({0.0, 0.0, 0.0, 0.0, 0.5}, 1.0, 2), std::invalid_argument);
}

}  // namespace basis_cyclic_bspline_test
