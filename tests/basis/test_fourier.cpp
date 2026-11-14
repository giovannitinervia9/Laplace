// Fourier bases (include/laplace/basis/fourier.hpp) against the mpmath
// references of tools/reference/basis_fourier.py: rows of order 0..6 and
// integrals evaluated in a Kokkos kernel on the default execution space, cross
// Gram matrices on the host. 27 configurations: six domains (full periods on
// [0, 1], [-3, 4.5] and [1000, 1001]; a period shorter than the width, [0, 10]
// with period 7; a longer one, [0, 9] with period 12; [1000, 1003] with period
// 0.7), H = 0, 1, 4, 20, 100 harmonics; points inside and outside the domain
// (the basis is periodic) and points where j u is near an integer or a
// half-integer (u = (x - lower)/period). H = 100 on three of the domains, and
// there only a subset of the columns is in the file (the others are skipped).
// Errors: rows and integrals in ulps of max(1, the row's largest |value|),
// also as a ratio to the conditioning max(1, 2 pi j |x'|/period) of the column
// (the error of the argument j nu x' caused by the rounding of x'/period) plus
// k for the order k (the rounding of nu and of the power (j nu)^k); Gram entries in
// ulps of the largest entry of the reference. Then: expansions against the
// row dot beta in a kernel, periodicity, the full-period Gram matrices
// (block diagonal), fill_basis_rows, the concept, the constructors.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/basis/fourier.hpp>

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

namespace basis_fourier_test {

namespace lb = laplace::basis;
static_assert(lb::univariate_basis<lb::fourier_basis>);

constexpr int HMAX = 100;
constexpr int W = 2 * HMAX + 1;  // widest row
constexpr int NQ = 8;            // orders 0..6, then the integral

struct case_data {
  std::string name;
  double lower = 0, period = 1, upper = 1;
  int H = 0;
  std::vector<double> x;
  std::vector<std::vector<double>> value;  // [point][quantity * W + i]
  std::map<std::pair<int, int>, std::vector<double>> gram;  // (k, l) -> K x K, column-major, NaN = not given
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
  std::string line;
  std::ifstream cf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_fourier_cases.csv");
  REQUIRE(cf.good());
  std::getline(cf, line);
  while (std::getline(cf, line)) {
    auto f = split(line);
    case_data c;
    c.name = f[1];
    c.lower = hex(f[2]);
    c.period = hex(f[3]);
    c.upper = hex(f[4]);
    c.H = std::stoi(f[5]);
    cs.push_back(c);
  }
  std::ifstream pf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_fourier_points.csv");
  REQUIRE(pf.good());
  std::getline(pf, line);
  while (std::getline(pf, line)) {
    auto f = split(line);
    auto& c = cs[std::stoi(f[0])];
    c.x.push_back(hex(f[2]));
    c.value.emplace_back(NQ * W, std::nan(""));
  }
  std::ifstream rf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_fourier.csv");
  REQUIRE(rf.good());
  std::getline(rf, line);
  while (std::getline(rf, line)) {
    auto f = split(line);
    auto& c = cs[std::stoi(f[0])];
    const int q = std::stoi(f[1]), k = std::stoi(f[2]), l = std::stoi(f[3]);
    const int p = std::stoi(f[4]), i = std::stoi(f[5]), j = std::stoi(f[6]);
    const double v = hex(f[7]);
    if (q == 0) {
      c.value[p][k * W + i] = v;
    } else if (q == 1) {
      c.value[p][7 * W + i] = v;
    } else {
      const int K = 2 * c.H + 1;
      auto& g = c.gram[{k, l}];
      if (g.empty()) g.assign(static_cast<size_t>(K) * K, std::nan(""));
      g[i + static_cast<size_t>(j) * K] = v;
    }
  }
  return cs;
}

using DV = Kokkos::View<double*>;
using DV2 = Kokkos::View<double**, Kokkos::LayoutRight>;

struct eval_functor {
  lb::fourier_basis b;
  DV x;
  DV2 out;
  KOKKOS_INLINE_FUNCTION void operator()(int m) const {
    double r[W];
    for (int k = 0; k < NQ; ++k) {
      if (k < 7) lb::basis_row_of_order(b, k, x(m), r);
      else b.integral_row(x(m), r);
      for (int q = 0; q < W; ++q) out(m, k * W + q) = q < b.row_width() ? r[q] : 0.0;
    }
  }
};

void eval_device(const lb::fourier_basis& b, const std::vector<double>& xs, std::vector<std::vector<double>>& out) {
  const int n = static_cast<int>(xs.size());
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int m = 0; m < n; ++m) hx(m) = xs[m];
  Kokkos::deep_copy(x, hx);
  DV2 o("out", n, NQ * W);
  Kokkos::parallel_for("fourier_eval", n, eval_functor{b, x, o});
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
  out.assign(n, std::vector<double>(NQ * W));
  for (int m = 0; m < n; ++m)
    for (int j = 0; j < NQ * W; ++j) out[m][j] = ho(m, j);
}

// the seven expansions and the integral expansion at every point, in a kernel
struct expansion_functor {
  lb::fourier_basis b;
  DV x, beta;
  DV2 out;
  KOKKOS_INLINE_FUNCTION void operator()(int m) const {
    const double* be = beta.data();
    out(m, 0) = b.expansion(x(m), be);
    out(m, 1) = b.expansion_d1(x(m), be);
    out(m, 2) = b.expansion_d2(x(m), be);
    out(m, 3) = b.expansion_d3(x(m), be);
    out(m, 4) = b.expansion_d4(x(m), be);
    out(m, 5) = b.expansion_d5(x(m), be);
    out(m, 6) = b.expansion_d6(x(m), be);
    out(m, 7) = b.integral_expansion(x(m), be);
  }
};

double ulp_of(double s) { return std::nextafter(s, INFINITY) - s; }
bool print_errors() { return std::getenv("LAPLACE_PRINT_ERRORS") != nullptr; }
double finite_or_inf(double e) { return e == e ? e : INFINITY; }

// bounds: measured maximum (gcc and CUDA, 2026-11-14) times 1.5. Measured rows
// (ulps, H = 0, 1, 4, 20, 100): 0, 24.5, 93, 437, 2147, all within 2.5 times the
// conditioning of the angle; integrals 2.0; Gram matrices 18.
struct bounds { double row, ratio, integral, gram; };
bounds bound_for(int H) {
  const double rows[5] = {0.5, 37.0, 140.0, 660.0, 3300.0};
  const int idx = H == 0 ? 0 : H <= 1 ? 1 : H <= 4 ? 2 : H <= 20 ? 3 : 4;
  return {rows[idx], 3.75, 3.0, 27.0};
}

struct table_row { double row = 0, ratio = 0, integral = 0, gram = 0; };
std::map<int, table_row> table;

void check_case(const case_data& c) {
  INFO("case " << c.name << " H=" << c.H);
  const int H = c.H, K = 2 * H + 1;
  const auto b = lb::make_fourier_basis(c.lower, c.period, H, c.upper);
  CHECK(b.dimension() == K);
  CHECK(b.row_width() == K);
  CHECK(b.scratch_size() == 0);

  std::vector<std::vector<double>> out;
  eval_device(b, c.x, out);

  double e_row = 0, e_ratio = 0, e_int = 0;
  const double pi2 = 6.283185307179586;
  for (size_t m = 0; m < c.x.size(); ++m) {
    const double xp = std::fabs((c.x[m] - c.lower) / c.period);
    for (int k = 0; k < NQ; ++k) {
      double scale = 1;
      for (int i = 0; i < K; ++i) {
        const double v = c.value[m][k * W + i];
        if (v == v) scale = std::max(scale, std::fabs(v));
      }
      for (int i = 0; i < K; ++i) {
        if (!(c.value[m][k * W + i] == c.value[m][k * W + i])) continue;  // column not in the file
        const double e = finite_or_inf(std::fabs(out[m][k * W + i] - c.value[m][k * W + i]) / ulp_of(scale));
        if (k == 7) {
          e_int = std::max(e_int, e);
        } else {
          e_row = std::max(e_row, e);
        }
        // the error of column j, in ulps of the row's scale, against the conditioning of its angle
        // (the amplitude (j nu)^k of a column never exceeds the row's scale)
        const int j = (i + 1) / 2;
        // plus k ulps, the rounding of nu and of the power (j nu)^k
        e_ratio = std::max(e_ratio, e / (std::max(1.0, pi2 * j * xp) + (k >= 1 && k <= 6 ? k : 0)));
      }
    }
    // the first column: the constant, exactly
    CHECK(out[m][0] == 1.0);
    for (int k = 1; k < 7; ++k) CHECK(out[m][k * W] == 0.0);
  }

  // Gram matrices on the host
  double e_gram = 0;
  std::vector<double> g(static_cast<size_t>(K) * K);
  for (const auto& [kl, ref] : c.gram) {
    b.cross_gram(kl.first, kl.second, g.data());
    double gm = 0;
    for (double v : ref)
      if (v == v) gm = std::max(gm, std::fabs(v));
    if (gm == 0) gm = 1;
    for (size_t t = 0; t < ref.size(); ++t)
      if (ref[t] == ref[t]) e_gram = std::max(e_gram, finite_or_inf(std::fabs(g[t] - ref[t]) / ulp_of(gm)));
  }

  auto& t = table[H];
  t.row = std::max(t.row, e_row);
  t.ratio = std::max(t.ratio, e_ratio);
  t.integral = std::max(t.integral, e_int);
  t.gram = std::max(t.gram, e_gram);
  if (print_errors())
    std::printf("H=%3d %-16s rows %6.1f (ratio %5.2f)  integral %6.1f  gram %6.1f\n", H, c.name.c_str(), e_row, e_ratio,
                e_int, e_gram);
  const bounds bd = bound_for(H);
  CHECK(e_row <= bd.row);
  CHECK(e_ratio <= bd.ratio);
  CHECK(e_int <= bd.integral);
  CHECK(e_gram <= bd.gram);
}

TEST_CASE("fourier: rows, integrals and Gram matrices against mpmath") {
  auto cs = read_cases();
  REQUIRE(cs.size() == 27);
  table.clear();
  for (const auto& c : cs) check_case(c);
  if (print_errors()) {
    std::printf("max over the six domains, by H (device rows and integrals; host Gram):\n");
    for (const auto& [H, t] : table)
      std::printf("H=%3d rows %6.1f ulp  ratio %5.2f  integral %6.1f ulp  gram %6.1f ulp\n", H, t.row, t.ratio,
                  t.integral, t.gram);
  }
}

TEST_CASE("fourier: expansions against row dot beta, in a kernel") {
  for (int H : {0, 1, 4, 20, 100}) {
    for (double lower : {0.0, -3.0, 1000.0}) {
      INFO("H=" << H << " lower=" << lower);
      const double period = 7.5;
      const auto b = lb::make_fourier_basis(lower, period, H, lower + 1.3 * period);
      const int K = b.dimension();
      std::vector<double> xs;
      for (int i = 0; i <= 40; ++i) xs.push_back(lower - 0.4 * period + 2.0 * period * i / 40.0);
      const int n = static_cast<int>(xs.size());
      DV x("x", n), beta("beta", K);
      auto hx = Kokkos::create_mirror_view(x);
      auto hb = Kokkos::create_mirror_view(beta);
      for (int m = 0; m < n; ++m) hx(m) = xs[m];
      for (int j = 0; j < K; ++j) hb(j) = std::sin(1.7 * j + 0.3) / (1.0 + 0.1 * j);
      Kokkos::deep_copy(x, hx);
      Kokkos::deep_copy(beta, hb);
      DV2 o("o", n, NQ);
      Kokkos::parallel_for("fourier_expansion", n, expansion_functor{b, x, beta, o});
      auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
      std::vector<double> r(K);
      for (int m = 0; m < n; ++m) {
        for (int k = 0; k < NQ; ++k) {
          if (k < 7) lb::basis_row_of_order(b, k, xs[m], r.data());
          else b.integral_row(xs[m], r.data());
          // the host row and the device expansion differ by the rounding of the
          // sum and of the device's sin and cos: a few ulps of sum |beta_j r_j|
          double dot = 0, absdot = 0;
          for (int j = 0; j < K; ++j) {
            dot += hb(j) * r[j];
            absdot += std::fabs(hb(j) * r[j]);
          }
          CHECK(std::fabs(ho(m, k) - dot) <= 8e-15 * absdot * (1.0 + 1e-3 * H) + 1e-300);
        }
      }
    }
  }
}

TEST_CASE("fourier: periodicity and the shift identity between orders") {
  const double period = 4.0;
  for (int H : {1, 4, 20}) {
    const auto b = lb::make_fourier_basis(-2.0, period, H);
    const int K = b.dimension();
    std::vector<double> r0(K), r1(K), rp(K), rm(K);
    for (int i = 0; i <= 40; ++i) {
      const double x = -2.0 + period * i / 40.0;
      for (int k = 0; k <= 6; ++k) {
        lb::basis_row_of_order(b, k, x, r0.data());
        lb::basis_row_of_order(b, k, x + 3 * period, r1.data());
        double scale = 1;
        for (double v : r0) scale = std::max(scale, std::fabs(v));
        for (int j = 0; j < K; ++j) CHECK(std::fabs(r0[j] - r1[j]) <= 3e-14 * (1.0 + 2 * 3.14159 * H * 4) * scale);
      }
      // derivative of order k + 1 is the central difference of order k (a loose check of the signs)
      const double h = 1e-5;
      for (int k = 0; k <= 5; ++k) {
        lb::basis_row_of_order(b, k, x + h, rp.data());
        lb::basis_row_of_order(b, k, x - h, rm.data());
        lb::basis_row_of_order(b, k + 1, x, r1.data());
        double scale = 1;
        for (double v : r1) scale = std::max(scale, std::fabs(v));
        for (int j = 0; j < K; ++j) CHECK(std::fabs((rp[j] - rm[j]) / (2 * h) - r1[j]) <= 1e-6 * scale * (1.0 + k));
      }
    }
  }
}

TEST_CASE("fourier: Gram matrices on a full period are block diagonal") {
  const auto b = lb::make_fourier_basis(2.0, 3.0, 5);
  const int K = b.dimension();
  CHECK(b.gram_bandwidth() == 1);
  CHECK(lb::make_fourier_basis(2.0, 3.0, 5, 7.0).gram_bandwidth() == 10);  // longer than the period
  CHECK(lb::make_fourier_basis(2.0, 3.0, 0).gram_bandwidth() == 0);
  std::vector<double> g(static_cast<size_t>(K) * K);
  for (int k = 0; k <= 6; ++k)
    for (int l = 0; l <= 6; ++l) {
      b.cross_gram(k, l, g.data());
      for (int j = 0; j < K; ++j)
        for (int i = 0; i < K; ++i) {
          const bool same_block = (i + 1) / 2 == (j + 1) / 2;
          if (!same_block) CHECK(g[i + static_cast<size_t>(j) * K] == 0.0);
        }
    }
  // k = l = 0: omega for the constant and omega/2 on the diagonal
  b.cross_gram(0, 0, g.data());
  CHECK(g[0] == 3.0);
  for (int i = 1; i < K; ++i) CHECK(g[i + static_cast<size_t>(i) * K] == doctest::Approx(1.5).epsilon(1e-15));
}

TEST_CASE("fourier: fill_basis_rows matches row") {
  const auto b = lb::make_fourier_basis(0.0, 1.0, 3);
  const int n = 101, K = b.dimension();
  DV x("x", n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) hx(i) = -0.3 + 1.6 * i / 100.0;
  Kokkos::deep_copy(x, hx);
  Kokkos::View<int*> first("first", n);
  DV2 values("values", n, K);
  lb::fill_basis_rows(b, 2, x, first, values);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), first);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::vector<double> r(K);
  for (int i = 0; i < n; ++i) {
    CHECK(b.row_d2(hx(i), r.data()) == hf(i));
    for (int q = 0; q < K; ++q) CHECK(r[q] == doctest::Approx(hv(i, q)).epsilon(1e-13).scale(100));
  }
}

TEST_CASE("fourier: constructors") {
  using lb::make_fourier_basis;
  const double inf = INFINITY, nan = std::nan("");
  CHECK_THROWS_AS(make_fourier_basis(0.0, 0.0, 3), std::invalid_argument);        // period 0
  CHECK_THROWS_AS(make_fourier_basis(0.0, -1.0, 3), std::invalid_argument);       // period < 0
  CHECK_THROWS_AS(make_fourier_basis(0.0, 1.0, -1), std::invalid_argument);       // harmonics < 0
  CHECK_THROWS_AS(make_fourier_basis(0.0, 1.0, 3, 0.0), std::invalid_argument);   // empty domain
  CHECK_THROWS_AS(make_fourier_basis(1.0, 1.0, 3, 0.5), std::invalid_argument);   // upper < lower
  CHECK_THROWS_AS(make_fourier_basis(nan, 1.0, 3), std::invalid_argument);
  CHECK_THROWS_AS(make_fourier_basis(0.0, inf, 3), std::invalid_argument);
  CHECK_THROWS_AS(make_fourier_basis(0.0, 1.0, 3, inf), std::invalid_argument);
  CHECK_NOTHROW(make_fourier_basis(0.0, 1.0, 0));
  const auto b = make_fourier_basis(-1.0, 2.0, 4);
  CHECK(b.dimension() == 9);
  CHECK(b.upper == 1.0);
  const auto c = make_fourier_basis(-1.0, 2.0, 4, 0.5);
  CHECK(c.upper == 0.5);
  CHECK(c.period == 2.0);
  double r[9];
  CHECK(b.row(std::nan(""), r) == 0);
  CHECK(std::isnan(r[1]));
  CHECK(r[0] == 1.0);
}

}  // namespace basis_fourier_test
