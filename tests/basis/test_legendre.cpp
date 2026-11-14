// The orthonormal Legendre basis (include/laplace/basis/legendre.hpp) against the
// exact references of tools/reference/basis_legendre.py: rows of order 0..6 and
// integrals evaluated in a Kokkos kernel on the default execution space, the cross
// Gram matrices on the host. n in {0, 1, 2, 5, 10, 20, 50} on [0, 1], [-1, 1],
// [-3, 7] and [1000, 1001]; random points, the two ends, points at 1e-k of the ends.
// Errors: rows and integrals in ulps of the row's largest |value| (also as a
// multiple of the conditioning of the evaluation in t: eps/2 (|b^(k)| + (1 + |t|)|b^(k+1)| w/2)),
// Gram entries in ulps of the largest entry.
// Then: G_00 = I, orthonormality by quadrature of the rows, the integral
// against the rows, expansions against row . beta in a kernel, the outside
// rules, the Gram matrices with more than 64 nodes (n = 100), the concept,
// fill_basis_rows and make_legendre_basis.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/basis/legendre.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

#define BOUND_ROW20 175, 60, 50, 45, 52, 30, 36
#define BOUND_ROW50 830, 440, 300, 175, 220, 205, 160
#define BOUND_INT20 53
#define BOUND_INT50 300
#define BOUND_GRAM20 126
#define BOUND_GRAM50 580
#define BOUND_RATIO 20

namespace basis_legendre_test {

namespace lb = laplace::basis;
constexpr int W = 51;   // widest row (n = 50)
constexpr int NK = 8;   // orders 0..7 in the file

static_assert(lb::univariate_basis<lb::legendre_basis>);

struct case_data {
  double lower = 0, upper = 1;
  int n = 0;
  std::vector<double> x;
  std::map<double, int> point;
  std::vector<std::vector<double>> row;       // [point][k * W + i], k = 0..7
  std::vector<std::vector<double>> integral;  // [point][i]
  std::map<std::pair<int, int>, std::vector<double>> gram;  // (k, l) -> K x K, row-major (i, j)
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
  std::ifstream cf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_legendre_cases.csv");
  REQUIRE(cf.good());
  std::string line;
  std::getline(cf, line);
  while (std::getline(cf, line)) {
    auto f = split(line);
    cs.emplace_back();
    cs.back().lower = hex(f[1]);
    cs.back().upper = hex(f[2]);
    cs.back().n = std::stoi(f[3]);
  }
  std::ifstream rf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_legendre.csv");
  REQUIRE(rf.good());
  std::getline(rf, line);
  while (std::getline(rf, line)) {
    auto f = split(line);
    auto& c = cs[std::stoi(f[0])];
    const int q = std::stoi(f[1]), k = std::stoi(f[2]), l = std::stoi(f[3]);
    const double x = hex(f[4]), v = hex(f[7]);
    const int i = std::stoi(f[5]), j = std::stoi(f[6]);
    if (q == 0 || q == 1) {
      auto it = c.point.find(x);
      int m;
      if (it == c.point.end()) {
        m = static_cast<int>(c.x.size());
        c.point[x] = m;
        c.x.push_back(x);
        c.row.emplace_back(NK * W, std::nan(""));
        c.integral.emplace_back(W, std::nan(""));
      } else {
        m = it->second;
      }
      if (q == 0) c.row[m][k * W + i] = v;
      else c.integral[m][i] = v;
    } else {
      auto& g = c.gram[{k, l}];
      if (g.empty()) g.assign(static_cast<size_t>(c.n + 1) * (c.n + 1), 0.0);
      g[static_cast<size_t>(i) * (c.n + 1) + j] = v;
    }
  }
  return cs;
}

using DV = Kokkos::View<double*>;
using DV2 = Kokkos::View<double**, Kokkos::LayoutRight>;

// rows of order 0..6 (slots 0..6), the integral (slot 7), at every point
struct eval_functor {
  lb::legendre_basis b;
  DV x;
  DV2 out;
  KOKKOS_INLINE_FUNCTION void operator()(int m) const {
    double r[W];
    for (int k = 0; k < 8; ++k) {
      const int f = k < 7 ? lb::basis_row_of_order(b, k, x(m), r) : b.integral_row(x(m), r);
      for (int q = 0; q < W; ++q) out(m, k * W + q) = (q < b.row_width() && f == 0) ? r[q] : 0.0;
    }
  }
};

void eval_device(const lb::legendre_basis& b, const std::vector<double>& xs, std::vector<std::vector<double>>& out) {
  const int n = static_cast<int>(xs.size());
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int m = 0; m < n; ++m) hx(m) = xs[m];
  Kokkos::deep_copy(x, hx);
  DV2 o("out", n, 8 * W);
  Kokkos::parallel_for("legendre_eval", n, eval_functor{b, x, o});
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
  out.assign(n, std::vector<double>(8 * W));
  for (int m = 0; m < n; ++m)
    for (int j = 0; j < 8 * W; ++j) out[m][j] = ho(m, j);
}

double ulp_of(double s) { return std::nextafter(s, INFINITY) - s; }
bool print_errors() { return std::getenv("LAPLACE_PRINT_ERRORS") != nullptr; }

// bounds in ulps: the measured maximum plus about half again (see the
// Measured lines; the first bound is for n <= 20, the second for n = 50).
double row_bound(int k, int n) {
  static const double b20[7] = {BOUND_ROW20};
  static const double b50[7] = {BOUND_ROW50};
  return n <= 20 ? b20[k] : b50[k];
}
double integral_bound(int n) { return n <= 20 ? BOUND_INT20 : BOUND_INT50; }
double gram_bound(int n) { return n <= 20 ? BOUND_GRAM20 : BOUND_GRAM50; }
double ratio_bound = BOUND_RATIO;

struct case_errors {
  double row[7] = {0}, ratio[7] = {0}, integral = 0, gram = 0;
};
std::map<int, case_errors> worst;  // by n, over the cases and both execution spaces of the run

void check_case(const case_data& c, int index) {
  INFO("case " << index << " n=" << c.n << " [" << c.lower << ", " << c.upper << "]");
  const int n = c.n, K = n + 1;
  const lb::legendre_basis b = lb::make_legendre_basis(c.lower, c.upper, n);
  const double w = c.upper - c.lower;
  std::vector<std::vector<double>> out;
  eval_device(b, c.x, out);

  case_errors e;
  for (size_t m = 0; m < c.x.size(); ++m) {
    const double t = (c.x[m] - c.lower) <= (c.upper - c.x[m]) ? 2 * (c.x[m] - c.lower) / w - 1 : 1 - 2 * (c.upper - c.x[m]) / w;
    for (int k = 0; k <= 6; ++k) {
      if (k > n) {  // above the degree: exact zeros
        for (int q = 0; q < K; ++q) CHECK(out[m][k * W + q] == 0.0);
        continue;
      }
      double scale = 0;
      for (int q = 0; q < K; ++q) scale = std::max(scale, std::fabs(c.row[m][k * W + q]));
      if (scale == 0) scale = 1;
      for (int q = 0; q < K; ++q) {
        const double err = std::fabs(out[m][k * W + q] - c.row[m][k * W + q]);
        e.row[k] = std::max(e.row[k], err / ulp_of(scale));
        if (k + 1 <= std::min(n, 7)) {
          const double cond = std::ldexp(1.0, -53) *
                              (std::fabs(c.row[m][k * W + q]) + (1 + std::fabs(t)) * std::fabs(c.row[m][(k + 1) * W + q]) * w / 2);
          if (cond > 0) e.ratio[k] = std::max(e.ratio[k], err / cond);
        }
      }
    }
    double scale = 0;
    for (int q = 0; q < K; ++q) scale = std::max(scale, std::fabs(c.integral[m][q]));
    if (scale == 0) scale = 1;
    for (int q = 0; q < K; ++q)
      e.integral = std::max(e.integral, std::fabs(out[m][7 * W + q] - c.integral[m][q]) / ulp_of(scale));
  }

  std::vector<double> g(static_cast<size_t>(K) * K);
  for (const auto& [kl, ref] : c.gram) {
    b.cross_gram(kl.first, kl.second, g.data());
    double gm = 0;
    for (double v : ref) gm = std::max(gm, std::fabs(v));
    if (gm == 0) gm = 1;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < K; ++j)
        e.gram = std::max(e.gram, std::fabs(g[i + static_cast<size_t>(j) * K] - ref[static_cast<size_t>(i) * K + j]) / ulp_of(gm));
    if (kl.first == 0 && kl.second == 0)
      for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j) CHECK(g[i + static_cast<size_t>(j) * K] == (i == j ? 1.0 : 0.0));
  }

  auto& wst = worst[n];
  for (int k = 0; k < 7; ++k) {
    wst.row[k] = std::max(wst.row[k], e.row[k]);
    wst.ratio[k] = std::max(wst.ratio[k], e.ratio[k]);
  }
  wst.integral = std::max(wst.integral, e.integral);
  wst.gram = std::max(wst.gram, e.gram);
  for (int k = 0; k <= std::min(n, 6); ++k) {
    CHECK(e.row[k] <= row_bound(k, n));
    CHECK(e.ratio[k] <= ratio_bound);
  }
  CHECK(e.integral <= integral_bound(n));
  CHECK(e.gram <= gram_bound(n));
}

TEST_CASE("legendre: rows, integrals and Gram matrices against exact references") {
  auto cs = read_cases();
  REQUIRE(cs.size() == 28);
  worst.clear();
  for (size_t i = 0; i < cs.size(); ++i) check_case(cs[i], static_cast<int>(i));
  if (print_errors()) {
    std::printf("%-3s %-7s %-7s %-7s %-7s %-7s %-7s %-7s | %-7s | %-7s   (ulps of the row's largest)\n", "n", "k=0", "k=1",
                "k=2", "k=3", "k=4", "k=5", "k=6", "integral", "gram");
    for (const auto& [n, e] : worst)
      std::printf("%-3d %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f | %7.1f | %7.1f\n", n, e.row[0], e.row[1], e.row[2],
                  e.row[3], e.row[4], e.row[5], e.row[6], e.integral, e.gram);
    std::printf("ratio to the conditioning (eps/2 (|b^(k)| + (1 + |t|)|b^(k+1)| w/2)):\n");
    for (const auto& [n, e] : worst)
      std::printf("%-3d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f\n", n, e.ratio[0], e.ratio[1], e.ratio[2], e.ratio[3],
                  e.ratio[4], e.ratio[5], e.ratio[6]);
  }
}

// ---------------------------------------------------------------------------
// expansions in a kernel: sum_m beta_m b_m^(k)(x), k = 0..6, and the integral

struct expansion_functor {
  lb::legendre_basis b;
  DV x, beta;
  DV2 out;
  KOKKOS_INLINE_FUNCTION void operator()(int m) const {
    const double* bt = beta.data();
    const double xm = x(m);
    out(m, 0) = b.expansion(xm, bt);
    out(m, 1) = b.expansion_d1(xm, bt);
    out(m, 2) = b.expansion_d2(xm, bt);
    out(m, 3) = b.expansion_d3(xm, bt);
    out(m, 4) = b.expansion_d4(xm, bt);
    out(m, 5) = b.expansion_d5(xm, bt);
    out(m, 6) = b.expansion_d6(xm, bt);
    out(m, 7) = b.integral_expansion(xm, bt);
  }
};

void expansions_device(const lb::legendre_basis& b, const std::vector<double>& xs, const std::vector<double>& beta,
                       std::vector<std::vector<double>>& out) {
  const int n = static_cast<int>(xs.size());
  DV x("x", n), bt("beta", static_cast<int>(beta.size()));
  auto hx = Kokkos::create_mirror_view(x);
  auto hb = Kokkos::create_mirror_view(bt);
  for (int m = 0; m < n; ++m) hx(m) = xs[m];
  for (size_t j = 0; j < beta.size(); ++j) hb(j) = beta[j];
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(bt, hb);
  DV2 o("out", n, 8);
  Kokkos::parallel_for("legendre_expansions", n, expansion_functor{b, x, bt, o});
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
  out.assign(n, std::vector<double>(8));
  for (int m = 0; m < n; ++m)
    for (int j = 0; j < 8; ++j) out[m][j] = ho(m, j);
}

TEST_CASE("legendre: expansions against row . beta, in a kernel") {
  for (int n : {0, 2, 10, 50}) {
    INFO("n=" << n);
    const auto b = lb::make_legendre_basis(-3.0, 7.0, n);
    const int K = n + 1;
    std::vector<double> beta(K), xs;
    for (int j = 0; j < K; ++j) beta[j] = std::sin(1.7 * j + 0.3) / (1 + j);
    for (int i = 0; i <= 40; ++i) xs.push_back(-5.0 + 14.0 * i / 40.0);  // also outside: linear rule
    std::vector<std::vector<double>> dev;
    expansions_device(b, xs, beta, dev);
    std::vector<double> r(K);
    double worst_ulp = 0;
    for (size_t m = 0; m < xs.size(); ++m) {
      for (int k = 0; k <= 6; ++k) {
        lb::basis_row_of_order(b, k, xs[m], r.data());
        double s = 0, sa = 0;
        for (int j = 0; j < K; ++j) { s += beta[j] * r[j]; sa += std::fabs(beta[j] * r[j]); }
        if (sa == 0) { CHECK(dev[m][k] == 0.0); continue; }
        worst_ulp = std::max(worst_ulp, std::fabs(dev[m][k] - s) / ulp_of(sa));
        CHECK(std::fabs(dev[m][k] - s) <= 64 * ulp_of(sa));
      }
      b.integral_row(xs[m], r.data());
      double s = 0, sa = 0;
      for (int j = 0; j < K; ++j) { s += beta[j] * r[j]; sa += std::fabs(beta[j] * r[j]); }
      worst_ulp = std::max(worst_ulp, std::fabs(dev[m][7] - s) / ulp_of(sa));
      CHECK(std::fabs(dev[m][7] - s) <= 64 * ulp_of(sa));
    }
    if (print_errors()) std::printf("expansions n=%2d: worst %.1f ulps of sum |beta b|\n", n, worst_ulp);
  }
}

// ---------------------------------------------------------------------------

TEST_CASE("legendre: orthonormality, the integral against the rows, the end values") {
  const double lo = -3.0, hi = 7.0, w = hi - lo;
  const int n = 20, K = n + 1;
  const auto b = lb::make_legendre_basis(lo, hi, n);
  // Gauss-Legendre with 64 nodes is exact for the degree 2n = 40
  std::vector<double> gram(static_cast<size_t>(K) * K, 0.0), r(K);
  for (int i = 0; i < 64; ++i) {
    const double t = laplace::math::quad::detail::legendre64(i), wq = laplace::math::quad::detail::legendre64(64 + i);
    const double x = lo + w * (t + 1) / 2;
    b.row(x, r.data());
    for (int p = 0; p < K; ++p)
      for (int q = 0; q < K; ++q) gram[p + static_cast<size_t>(q) * K] += wq * (w / 2) * r[p] * r[q];
  }
  for (int p = 0; p < K; ++p)
    for (int q = 0; q < K; ++q) CHECK(gram[p + static_cast<size_t>(q) * K] == doctest::Approx(p == q ? 1.0 : 0.0).epsilon(1e-13).scale(1));

  // d/dx integral_row = row (central differences of the closed form)
  std::vector<double> ip(K), im(K), rr(K);
  const double h = 1e-5;
  for (double x : {-2.9, -1.0, 0.3, 2.2, 6.8}) {
    b.integral_row(x + h, ip.data());
    b.integral_row(x - h, im.data());
    b.row(x, rr.data());
    for (int m = 0; m < K; ++m) CHECK((ip[m] - im[m]) / (2 * h) == doctest::Approx(rr[m]).epsilon(1e-7).scale(1));
  }
  // the integral over the whole interval: column 0 gives sqrt(w), the others 0 (orthogonal to 1) exactly
  b.integral_row(hi, ip.data());
  CHECK(ip[0] == doctest::Approx(std::sqrt(w)).epsilon(1e-15));
  for (int m = 1; m < K; ++m) CHECK(ip[m] == 0.0);
  b.integral_row(lo, ip.data());
  for (int m = 0; m < K; ++m) CHECK(ip[m] == 0.0);
  // end values: b_m(upper) = sqrt((2 m + 1)/w), b_m(lower) = (-1)^m times it
  b.row(hi, rr.data());
  for (int m = 0; m < K; ++m) CHECK(rr[m] == doctest::Approx(std::sqrt((2 * m + 1) / w)).epsilon(1e-14));
  b.row(lo, rr.data());
  for (int m = 0; m < K; ++m) CHECK(rr[m] == doctest::Approx((m % 2 ? -1 : 1) * std::sqrt((2 * m + 1) / w)).epsilon(1e-14));
}

TEST_CASE("legendre: Gram matrices with more than 64 nodes (n = 100)") {
  // G_10 + G_01 = the matrix of products at the ends, b_i(upper) b_j(upper) - b_i(lower) b_j(lower):
  // integration by parts, independent of the references. For n = 100 the integrand of G_10 has degree
  // 199 and cross_gram uses a rule with 100 nodes.
  const int n = 100, K = n + 1;
  for (auto [lo, hi] : {std::pair<double, double>{0.0, 1.0}, std::pair<double, double>{-3.0, 7.0}}) {
    const auto b = lb::make_legendre_basis(lo, hi, n);
    std::vector<double> g10(static_cast<size_t>(K) * K), g01(static_cast<size_t>(K) * K), a(K), c(K);
    b.cross_gram(1, 0, g10.data());
    b.cross_gram(0, 1, g01.data());
    b.row(hi, a.data());
    b.row(lo, c.data());
    double gm = 0, worst_diff = 0;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < K; ++j) {
        const size_t id = i + static_cast<size_t>(j) * K;
        gm = std::max(gm, std::fabs(g10[id]));
        worst_diff = std::max(worst_diff, std::fabs(g10[id] + g01[id] - (a[i] * a[j] - c[i] * c[j])));
      }
    if (print_errors()) std::printf("large-n rule n=100 [%g, %g]: |G10 + G01 - ends| / max |G10| = %.2e\n", lo, hi, worst_diff / gm);
    CHECK(worst_diff <= 5e-13 * gm);
    // G_11 is symmetric positive: the diagonal is positive and the matrix symmetric to rounding
    std::vector<double> g11(static_cast<size_t>(K) * K);
    b.cross_gram(1, 1, g11.data());
    double gm1 = 0, asym = 0;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < K; ++j) {
        gm1 = std::max(gm1, std::fabs(g11[i + static_cast<size_t>(j) * K]));
        asym = std::max(asym, std::fabs(g11[i + static_cast<size_t>(j) * K] - g11[j + static_cast<size_t>(i) * K]));
      }
    CHECK(asym <= 1e-13 * gm1);
    for (int i = 1; i < K; ++i) CHECK(g11[i + static_cast<size_t>(i) * K] > 0.0);
    CHECK(g11[0] == 0.0);  // b_0' = 0
  }
}

TEST_CASE("legendre: outside the domain") {
  using rule = lb::outside_domain_rule;
  const double xs[2] = {-0.2, 1.3};
  for (int n : {0, 1, 4, 12}) {
    INFO("n=" << n);
    const int K = n + 1;
    const auto lin = lb::make_legendre_basis(0.0, 1.0, n, rule::linear);
    std::vector<double> r(K), v(K), g(K), ie(K), ix(K);
    for (double x : xs) {
      const double e = x < 0 ? 0.0 : 1.0, dx = x - e;
      CHECK(lin.row(x, r.data()) == 0);
      lin.row(e, v.data());
      lin.row_d1(e, g.data());
      for (int q = 0; q < K; ++q) CHECK(r[q] == doctest::Approx(v[q] + g[q] * dx).epsilon(1e-14).scale(1));
      lin.row_d1(x, r.data());
      for (int q = 0; q < K; ++q) CHECK(r[q] == g[q]);
      for (int k = 2; k <= 6; ++k) {
        lb::basis_row_of_order(lin, k, x, r.data());
        for (int q = 0; q < K; ++q) CHECK(r[q] == 0.0);
      }
      lin.integral_row(e, ie.data());
      lin.integral_row(x, ix.data());
      for (int q = 0; q < K; ++q) CHECK(ix[q] == doctest::Approx(ie[q] + v[q] * dx + g[q] * dx * dx / 2).epsilon(1e-14).scale(1));
      // expansions follow the same rule
      std::vector<double> beta(K);
      for (int j = 0; j < K; ++j) beta[j] = std::cos(0.9 * j + 0.2);
      lin.row(x, r.data());
      double s = 0, s1 = 0, si = 0;
      for (int q = 0; q < K; ++q) s += beta[q] * r[q];
      CHECK(lin.expansion(x, beta.data()) == doctest::Approx(s).epsilon(1e-14).scale(1));
      lin.row_d1(x, r.data());
      for (int q = 0; q < K; ++q) s1 += beta[q] * r[q];
      CHECK(lin.expansion_d1(x, beta.data()) == doctest::Approx(s1).epsilon(1e-14).scale(1));
      CHECK(lin.expansion_d3(x, beta.data()) == 0.0);
      lin.integral_row(x, r.data());
      for (int q = 0; q < K; ++q) si += beta[q] * r[q];
      CHECK(lin.integral_expansion(x, beta.data()) == doctest::Approx(si).epsilon(1e-14).scale(1));
    }
    // polynomial: the polynomial continued (its derivative is the next order)
    const auto pol = lb::make_legendre_basis(0.0, 1.0, n, rule::polynomial);
    for (double x : xs) {
      const double h = 1e-6;
      std::vector<double> rp(K), rm(K);
      pol.row(x + h, rp.data());
      pol.row(x - h, rm.data());
      pol.row_d1(x, r.data());
      for (int q = 0; q < K; ++q) CHECK(r[q] == doctest::Approx((rp[q] - rm[q]) / (2 * h)).epsilon(1e-6).scale(1 + n * n * n));
    }
    // the polynomial rule agrees with the linear one inside the domain
    pol.row(0.37, r.data());
    lin.row(0.37, v.data());
    for (int q = 0; q < K; ++q) CHECK(r[q] == v[q]);
    // nan: NaN beyond 1e-8 of the width, the polynomial inside it
    const auto nn = lb::make_legendre_basis(0.0, 1.0, n, rule::nan);
    nn.row(1.3, r.data());
    for (int q = 0; q < K; ++q) CHECK(std::isnan(r[q]));
    nn.integral_row(-0.2, r.data());
    for (int q = 0; q < K; ++q) CHECK(std::isnan(r[q]));
    std::vector<double> beta(K, 1.0);
    CHECK(std::isnan(nn.expansion(1.3, beta.data())));
    CHECK(std::isnan(nn.integral_expansion(-0.2, beta.data())));
    nn.row(1.0 + 1e-10, r.data());
    pol.row(1.0 + 1e-10, v.data());
    for (int q = 0; q < K; ++q) CHECK(r[q] == v[q]);
    // NaN input
    lin.row(std::nan(""), r.data());
    CHECK(std::isnan(r[0]));
    CHECK(std::isnan(lin.expansion_d2(std::nan(""), beta.data())));
    CHECK(std::isnan(lin.integral_expansion(std::nan(""), beta.data())));
  }
}

TEST_CASE("legendre: fill_basis_rows matches row_d1") {
  const auto b = lb::make_legendre_basis(0.0, 1.0, 5);
  const int n = 101, K = b.row_width();
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) hx(i) = i / 100.0;
  Kokkos::deep_copy(x, hx);
  Kokkos::View<int*> first("first", n);
  DV2 values("values", n, K);
  lb::fill_basis_rows(b, 1, x, first, values);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), first);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::vector<double> r(K);
  for (int i = 0; i < n; ++i) {
    CHECK(b.row_d1(hx(i), r.data()) == hf(i));
    for (int q = 0; q < K; ++q) CHECK(r[q] == doctest::Approx(hv(i, q)).epsilon(1e-14).scale(1));
  }
}

TEST_CASE("legendre: make_legendre_basis checks its arguments") {
  const double inf = std::numeric_limits<double>::infinity(), nan = std::nan("");
  CHECK_THROWS_AS(lb::make_legendre_basis(1.0, 1.0, 3), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_legendre_basis(2.0, 1.0, 3), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_legendre_basis(nan, 1.0, 3), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_legendre_basis(0.0, inf, 3), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_legendre_basis(-inf, 1.0, 3), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_legendre_basis(0.0, 1.0, -1), std::invalid_argument);
  CHECK_NOTHROW(lb::make_legendre_basis(0.0, 1.0, 0));
  const auto b = lb::make_legendre_basis(-1.0, 3.0, 7, lb::outside_domain_rule::nan);
  CHECK(b.dimension() == 8);
  CHECK(b.row_width() == 8);
  CHECK(b.scratch_size() == 0);
  CHECK(b.gram_bandwidth() == 7);
  CHECK(b.outside == lb::outside_domain_rule::nan);
  CHECK(b.lower == -1.0);
  CHECK(b.upper == 3.0);
  CHECK(b.max_degree == 7);
}

}  // namespace basis_legendre_test
