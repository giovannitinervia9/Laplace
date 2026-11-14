// B-spline bases (include/laplace/basis/bspline.hpp) against the exact
// rational references of tools/reference/basis_bspline.py: rows of order
// 0..min(d, 6) and integrals evaluated in a Kokkos kernel on the default
// execution space, column totals and cross Gram matrices on the host. 42 knot
// configurations, degrees 0, 1, 2, 3, 5, 10: uniform clamped and extended,
// random, interior multiplicity 2..d + 1, an interval of width 1e-12, the
// domain [1000, 1001]; points on the knots and one ulp away.
// Errors: rows in ulps of the row's largest value; integrals and totals in
// ulps of the largest column total; Gram entries in ulps of the largest entry.
// Then: the arithmetic interval against the binary search, the outside
// rules, expansions, partition of unity, linear precision of the Greville
// abscissae, regularity points, constructors, quantile_knots.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/basis/bspline.hpp>

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

namespace basis_bspline_test {

namespace lb = laplace::basis;
constexpr int W = lb::bspline_max_degree + 1;  // widest row
constexpr int NQ = 8;                          // orders 0..6, then the integral

struct case_data {
  std::string name;
  int degree = 0, dim = 0, kind = 0;
  std::vector<double> knots;
  std::vector<double> x;                      // points, in file order
  std::map<double, int> point;                // x -> index
  std::vector<std::vector<int>> first;        // [point][quantity]
  std::vector<std::vector<double>> value;     // [point][quantity * W + i]
  std::vector<double> total;                  // [j]
  std::map<std::pair<int, int>, std::vector<double>> gram;  // (k, l) -> K x K, row-major
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
  std::ifstream kf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_bspline_knots.csv");
  REQUIRE(kf.good());
  std::string line;
  std::getline(kf, line);
  while (std::getline(kf, line)) {
    auto f = split(line);
    const int c = std::stoi(f[0]);
    if (c >= static_cast<int>(cs.size())) {
      cs.emplace_back();
      cs.back().name = f[1];
      cs.back().degree = std::stoi(f[2]);
      cs.back().dim = std::stoi(f[3]);
      cs.back().kind = std::stoi(f[4]);
    }
    cs[c].knots.push_back(hex(f[6]));
  }
  std::ifstream rf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_bspline.csv");
  REQUIRE(rf.good());
  std::getline(rf, line);
  while (std::getline(rf, line)) {
    auto f = split(line);
    auto& c = cs[std::stoi(f[0])];
    const int q = std::stoi(f[1]), k = std::stoi(f[2]), l = std::stoi(f[3]);
    const double x = hex(f[4]), v = hex(f[7]);
    const int first = std::stoi(f[5]), i = std::stoi(f[6]);
    if (q == 0 || q == 1) {
      auto it = c.point.find(x);
      int m;
      if (it == c.point.end()) {
        m = static_cast<int>(c.x.size());
        c.point[x] = m;
        c.x.push_back(x);
        c.first.emplace_back(NQ, -1);
        c.value.emplace_back(NQ * W, std::nan(""));
      } else {
        m = it->second;
      }
      const int qq = q == 0 ? k : 7;
      c.first[m][qq] = first;
      c.value[m][qq * W + i] = v;
    } else if (q == 2) {
      c.total.push_back(v);
    } else {
      auto& g = c.gram[{k, l}];
      if (g.empty()) g.assign(static_cast<size_t>(c.dim) * c.dim, 0.0);
      g[static_cast<size_t>(first) * c.dim + i] = v;
    }
  }
  return cs;
}

lb::bspline_basis_owner make_owner(const case_data& c) {
  const int d = c.degree, K = c.dim;
  if (c.kind == 0) return lb::bspline_basis_owner::uniform_clamped(c.knots[d], c.knots[K], K, d);
  if (c.kind == 1) return lb::bspline_basis_owner::uniform_extended(c.knots[d], c.knots[K], K, d);
  return lb::bspline_basis_owner::from_knots(c.knots, d);
}

using DV = Kokkos::View<double*>;
using DV2 = Kokkos::View<double**, Kokkos::LayoutRight>;
using IV2 = Kokkos::View<int**, Kokkos::LayoutRight>;

struct eval_functor {
  lb::bspline_basis b;
  DV x;
  DV2 out;
  IV2 first;
  KOKKOS_INLINE_FUNCTION void operator()(int m) const {
    double r[W];
    for (int k = 0; k < NQ; ++k) {
      first(m, k) = k < 7 ? lb::basis_row_of_order(b, k, x(m), r) : b.integral_row(x(m), r);
      for (int q = 0; q < W; ++q) out(m, k * W + q) = q < b.row_width() ? r[q] : 0.0;
    }
  }
};

void eval_device(const lb::bspline_basis& b, const std::vector<double>& xs, std::vector<std::vector<double>>& out,
                 std::vector<std::vector<int>>& first) {
  const int n = static_cast<int>(xs.size());
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int m = 0; m < n; ++m) hx(m) = xs[m];
  Kokkos::deep_copy(x, hx);
  DV2 o("out", n, NQ * W);
  IV2 f("first", n, NQ);
  Kokkos::parallel_for("bspline_eval", n, eval_functor{b, x, o, f});
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  out.assign(n, std::vector<double>(NQ * W));
  first.assign(n, std::vector<int>(NQ));
  for (int m = 0; m < n; ++m) {
    for (int j = 0; j < NQ * W; ++j) out[m][j] = ho(m, j);
    for (int k = 0; k < NQ; ++k) first[m][k] = hf(m, k);
  }
}

double ulp_of(double s) { return std::nextafter(s, INFINITY) - s; }
bool print_errors() { return std::getenv("LAPLACE_PRINT_ERRORS") != nullptr; }

// bounds in ulps: measured maximum (gcc, 2026-11-14) plus a margin, by degree.
// Measured: rows 7 (d <= 5) and 19 (d = 10); integrals and totals 3 and 4.7;
// Gram matrices 6 and 6.
double row_bound(int d) { return d <= 5 ? 12.0 : 32.0; }
double integral_bound(int d) { return d <= 5 ? 8.0 : 12.0; }
double gram_bound(int d) { return d <= 5 ? 12.0 : 12.0; }

void check_case(const case_data& c) {
  INFO("case " << c.name << " d=" << c.degree);
  const int d = c.degree, K = c.dim;
  auto owner = make_owner(c);
  // the constructors give the knots of the reference file
  REQUIRE(owner.knots().size() == c.knots.size());
  for (size_t j = 0; j < c.knots.size(); ++j) CHECK(owner.knots()[j] == c.knots[j]);

  std::vector<std::vector<double>> out;
  std::vector<std::vector<int>> first;
  eval_device(owner.device(), c.x, out, first);

  double tmax = 0;
  for (double v : c.total) tmax = std::max(tmax, std::fabs(v));
  double e_row[7] = {0}, e_int = 0;
  int bad_first = 0;
  for (size_t m = 0; m < c.x.size(); ++m) {
    for (int k = 0; k <= 6; ++k) {
      const bool has_ref = k <= std::min(d, 6);
      if (!has_ref) {  // above the degree: exact zeros
        for (int q = 0; q <= d; ++q) CHECK(out[m][k * W + q] == 0.0);
        continue;
      }
      if (first[m][k] != c.first[m][k]) { ++bad_first; continue; }
      double scale = 0;
      for (int q = 0; q <= d; ++q) scale = std::max(scale, std::fabs(c.value[m][k * W + q]));
      if (scale == 0) scale = 1;
      for (int q = 0; q <= d; ++q)
        e_row[k] = std::max(e_row[k], std::fabs(out[m][k * W + q] - c.value[m][k * W + q]) / ulp_of(scale));
    }
    if (first[m][7] != c.first[m][7]) { ++bad_first; continue; }
    for (int q = 0; q <= d; ++q)
      e_int = std::max(e_int, std::fabs(out[m][7 * W + q] - c.value[m][7 * W + q]) / ulp_of(tmax));
  }
  CHECK(bad_first == 0);

  // totals and Gram matrices on the host
  const auto hb = owner.host();
  double e_tot = 0;
  for (int j = 0; j < K; ++j) e_tot = std::max(e_tot, std::fabs(hb.integral_column_total(j) - c.total[j]) / ulp_of(tmax));
  double e_gram = 0;
  std::vector<double> g(static_cast<size_t>(K) * K);
  for (const auto& [kl, ref] : c.gram) {
    hb.cross_gram(kl.first, kl.second, g.data());
    double gm = 0;
    for (double v : ref) gm = std::max(gm, std::fabs(v));
    if (gm == 0) gm = 1;
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b)
        e_gram = std::max(e_gram, std::fabs(g[a + static_cast<size_t>(b) * K] - ref[static_cast<size_t>(a) * K + b]) / ulp_of(gm));
  }

  double e_rows = 0;
  for (double e : e_row) e_rows = std::max(e_rows, e);
  if (print_errors())
    std::printf("d=%2d %-17s rows k=0..6: %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f  integral %5.1f  total %5.1f  gram %5.1f\n",
                d, c.name.c_str(), e_row[0], e_row[1], e_row[2], e_row[3], e_row[4], e_row[5], e_row[6], e_int, e_tot,
                e_gram);
  CHECK(e_rows <= row_bound(d));
  CHECK(e_int <= integral_bound(d));
  CHECK(e_tot <= integral_bound(d));
  CHECK(e_gram <= gram_bound(d));

  // the arithmetic interval gives exactly the rows of the binary search
  if (hb.step > 0.0) {
    auto bs = hb;
    bs.step = 0.0;
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

TEST_CASE("bspline: rows, integrals, totals and Gram matrices against exact references") {
  auto cs = read_cases();
  REQUIRE(cs.size() == 42);
  for (const auto& c : cs) check_case(c);
}

TEST_CASE("bspline: outside the domain") {
  using rule = lb::outside_domain_rule;
  const double xs[2] = {-0.2, 1.3};
  for (int d : {1, 3, 5}) {
    INFO("d=" << d);
    const auto lin_owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 8, d, rule::linear);
    const auto lin = lin_owner.host();
    double r[W], v[W], g[W], ie[W], ix[W];
    for (double x : xs) {
      const double e = x < 0 ? 0.0 : 1.0, dx = x - e;
      const int f = lin.row(x, r);
      CHECK(lin.row(e, v) == f);
      lin.row_d1(e, g);
      for (int q = 0; q <= d; ++q) CHECK(r[q] == doctest::Approx(v[q] + g[q] * dx).epsilon(1e-15));
      lin.row_d1(x, r);
      for (int q = 0; q <= d; ++q) CHECK(r[q] == g[q]);
      lin.row_d2(x, r);
      for (int q = 0; q <= d; ++q) CHECK(r[q] == 0.0);
      lin.integral_row(e, ie);
      lin.integral_row(x, ix);
      for (int q = 0; q <= d; ++q)
        CHECK(ix[q] == doctest::Approx(ie[q] + v[q] * dx + g[q] * dx * dx / 2).epsilon(1e-14));
    }
    // polynomial: the end piece continued (its derivative is the next order)
    const auto pol_owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 8, d, rule::polynomial);
    const auto pol = pol_owner.host();
    for (double x : xs) {
      const double h = 1e-6;
      double rp[W], rm[W];
      pol.row(x + h, rp);
      pol.row(x - h, rm);
      pol.row_d1(x, r);
      for (int q = 0; q <= d; ++q) CHECK(r[q] == doctest::Approx((rp[q] - rm[q]) / (2 * h)).epsilon(1e-6).scale(1));
    }
    // nan: NaN beyond 1e-8 of the width, the end piece inside it
    const auto nan_owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 8, d, rule::nan);
    const auto nan = nan_owner.host();
    nan.row(1.3, r);
    for (int q = 0; q <= d; ++q) CHECK(std::isnan(r[q]));
    nan.integral_row(-0.2, r);
    for (int q = 0; q <= d; ++q) CHECK(std::isnan(r[q]));
    nan.row(1.0 + 1e-10, r);
    lin.row(1.0, v);
    for (int q = 0; q <= d; ++q) CHECK(r[q] == doctest::Approx(v[q]).epsilon(1e-8).scale(1));
    // NaN input
    lin.row(std::nan(""), r);
    CHECK(std::isnan(r[0]));
  }
}

TEST_CASE("bspline: expansions, partition of unity, Greville abscissae, regularity points") {
  auto owner = lb::bspline_basis_owner::clamped(0.0, 2.0, {0.3, 0.7, 0.7, 1.1, 1.1, 1.1, 1.6}, 3);
  const auto b = owner.host();
  const int K = b.dimension();
  std::vector<double> beta(K), xi(K);
  for (int j = 0; j < K; ++j) beta[j] = std::sin(1.7 * j + 0.3);
  b.greville(xi.data());
  double r[W], ir[W];
  for (int i = 0; i <= 200; ++i) {
    const double x = -0.3 + 2.6 * i / 200.0;  // also outside: linear rule
    const int f = b.row(x, r);
    double s = 0, sb = 0, sx = 0;
    for (int q = 0; q <= 3; ++q) { s += r[q]; sb += beta[f + q] * r[q]; sx += xi[f + q] * r[q]; }
    CHECK(s == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(sx == doctest::Approx(x).epsilon(1e-14).scale(1));  // linear precision
    CHECK(b.expansion(x, beta.data()) == doctest::Approx(sb).epsilon(1e-15).scale(1));
    const int f1 = b.row_d2(x, r);
    double s2 = 0;
    for (int q = 0; q <= 3; ++q) s2 += beta[f1 + q] * r[q];
    CHECK(b.expansion_d2(x, beta.data()) == doctest::Approx(s2).epsilon(1e-15).scale(1));
    const int fi = b.integral_row(x, ir);
    double si = 0;
    for (int j = 0; j < fi; ++j) si += beta[j] * b.integral_column_total(j);
    for (int q = 0; q <= 3; ++q) si += beta[fi + q] * ir[q];
    CHECK(b.integral_expansion(x, beta.data()) == doctest::Approx(si).epsilon(1e-14).scale(1));
  }
  const auto rp = b.regularity_points();
  // ends (linear rule, d = 3), 0.3 (C^2), 0.7 (C^1), 1.1 (C^0), 1.6 (C^2)
  REQUIRE(rp.size() == 6);
  CHECK(rp[0].x == 0.0); CHECK(rp[0].continuity == 1);
  CHECK(rp[1].x == 0.3); CHECK(rp[1].continuity == 2);
  CHECK(rp[2].x == 0.7); CHECK(rp[2].continuity == 1);
  CHECK(rp[3].x == 1.1); CHECK(rp[3].continuity == 0);
  CHECK(rp[4].x == 1.6); CHECK(rp[4].continuity == 2);
  CHECK(rp[5].x == 2.0); CHECK(rp[5].continuity == 1);
}

TEST_CASE("bspline: fill_basis_rows matches row") {
  auto owner = lb::bspline_basis_owner::uniform_extended(0.0, 1.0, 10, 3);
  const int n = 101;
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) hx(i) = i / 100.0;
  Kokkos::deep_copy(x, hx);
  Kokkos::View<int*> first("first", n);
  DV2 values("values", n, 4);
  lb::fill_basis_rows(owner.device(), 1, x, first, values);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), first);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  const auto b = owner.host();
  double r[4];
  for (int i = 0; i < n; ++i) {
    CHECK(b.row_d1(hx(i), r) == hf(i));
    for (int q = 0; q < 4; ++q) CHECK(r[q] == doctest::Approx(hv(i, q)).epsilon(1e-15).scale(1));
  }
}

TEST_CASE("bspline: constructors and quantile_knots") {
  using O = lb::bspline_basis_owner;
  CHECK_THROWS_AS(O::uniform_clamped(0.0, 1.0, 3, 3), std::invalid_argument);   // dim < d + 1
  CHECK_THROWS_AS(O::uniform_clamped(1.0, 1.0, 8, 3), std::invalid_argument);   // empty domain
  CHECK_THROWS_AS(O::uniform_clamped(0.0, 1.0, 20, 11), std::invalid_argument); // degree > 10
  CHECK_THROWS_AS(O::clamped(0.0, 1.0, {0.5, 0.5, 0.5, 0.5, 0.5}, 3), std::invalid_argument);  // mu > d + 1
  CHECK_THROWS_AS(O::clamped(0.0, 1.0, {0.7, 0.3}, 3), std::invalid_argument);  // decreasing
  CHECK_THROWS_AS(O::clamped(0.0, 1.0, {1.0}, 3), std::invalid_argument);       // on the end
  CHECK_NOTHROW(O::clamped(0.0, 1.0, {0.5, 0.5, 0.5, 0.5}, 3));                  // mu = d + 1: a jump
  auto o = O::uniform_clamped(0.0, 1.0, 8, 3);
  CHECK(o.host().dimension() == 8);
  CHECK(o.host().row_width() == 4);
  CHECK(o.host().gram_bandwidth() == 3);

  std::vector<double> x;
  for (int i = 1; i <= 100; ++i) x.push_back(i);
  auto k = lb::quantile_knots(x, 3);
  REQUIRE(k.size() == 3);
  CHECK(k[0] == doctest::Approx(25.75));
  CHECK(k[1] == doctest::Approx(50.5));
  CHECK(k[2] == doctest::Approx(75.25));
  std::vector<double> ties(50, 1.0);
  for (int i = 0; i < 50; ++i) ties.push_back(2.0 + i);
  CHECK_THROWS_AS(lb::quantile_knots(ties, 3), std::invalid_argument);
}

}  // namespace basis_bspline_test
