// dirichlet_distrib and multinomial_distrib (note/distrib_multivariate.md section 18):
//  - both satisfy multivariate_distrib (the Dirichlet continuous, the multinomial
//    discrete);
//  - against the mpmath references (tools/reference/distrib_simplex.py) and the
//    frozen statmodels7 results (tools/reference/statmodels7_simplex.R), evaluated in
//    a Kokkos kernel on the default execution space: blocks of the log-density
//    (log-mass) in eta and phi to order 6, the Dirichlet's derivatives in y, the
//    expected information and its derivatives to order 4, normwise in ulps of
//    max(max_entries |ref|, 1) (ulp1N), worst over the points of each group;
//  - draws (mean, covariance, on the simplex or summing to the size; finite at a
//    tiny Dirichlet shape), moments and start values.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/distrib/dirichlet.hpp>
#include <laplace/distrib/multinomial.hpp>

#include <Kokkos_Core.hpp>

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

namespace simplex_test {

namespace ld = laplace::distrib;

static_assert(ld::multivariate_distrib<ld::dirichlet_distrib>);
static_assert(ld::multivariate_distrib<ld::multinomial_distrib>);
static_assert(ld::dirichlet_distrib::kind == ld::distrib_kind::continuous);
static_assert(ld::multinomial_distrib::kind == ld::distrib_kind::discrete);

constexpr int max_d = 5, max_out = 512;

// one block: the rows of a reference file that share every column but entry, hi, lo
struct Block {
  int law, d, kind, J, A, K;
  double x[max_d - 1] = {}, phi = 0.0, y[max_d] = {}, n = 0.0;
  std::vector<int> pos;
  std::vector<double> hi, lo;
};

std::vector<double> hex_list(const std::string& s) {
  std::vector<double> v;
  if (s == "-") return v;
  std::stringstream ss(s);
  std::string f;
  while (std::getline(ss, f, ';')) v.push_back(std::strtod(f.c_str(), nullptr));
  return v;
}

std::vector<Block> read_blocks(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::string, int> where;
  std::vector<Block> blocks;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string t;
    while (std::getline(ss, t, ',')) f.push_back(t);
    REQUIRE(f.size() == 13);
    const std::string key = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8] + f[9];
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(blocks.size())).first;
      Block b;
      b.law = std::atoi(f[0].c_str()), b.d = std::atoi(f[1].c_str()), b.kind = std::atoi(f[6].c_str());
      b.J = std::atoi(f[7].c_str()), b.A = std::atoi(f[8].c_str()), b.K = int(hex_list(f[9]).size());
      const auto x = hex_list(f[2]), y = hex_list(f[4]);
      for (std::size_t q = 0; q < x.size(); ++q) b.x[q] = x[q];
      for (std::size_t q = 0; q < y.size(); ++q) b.y[q] = y[q];
      if (b.law == 1) b.phi = std::strtod(f[3].c_str(), nullptr);
      else b.n = std::strtod(f[5].c_str(), nullptr);
      blocks.push_back(b);
    }
    Block& b = blocks[it->second];
    b.pos.push_back(std::atoi(f[10].c_str()));
    b.hi.push_back(std::strtod(f[11].c_str(), nullptr));
    b.lo.push_back(std::strtod(f[12].c_str(), nullptr));
  }
  return blocks;
}

// the device side: one block per index
struct Eval {
  Kokkos::View<int* [6]> meta;  // law, d, kind, J, A, K
  // rows are read through pointers, so the layout is LayoutRight on every backend
  Kokkos::View<double* [2 * max_d + 1], Kokkos::LayoutRight> par;  // x (max_d - 1), phi, y (max_d), n
  Kokkos::View<double* [max_out], Kokkos::LayoutRight> out;
  Kokkos::View<double**, Kokkos::LayoutRight> work;

  template <int J, int K, int A>
  KOKKOS_INLINE_FUNCTION void run(int i) const {
    const int law = meta(i, 0), d = meta(i, 1), kind = meta(i, 2);
    const double* x = &par(i, 0);
    const double* y = &par(i, max_d);
    double* o = &out(i, 0);
    double* w = &work(i, 0);
    if (law == 1) {
      const ld::dirichlet_distrib D(d);
      const auto st = D.theta_state(&par(i, max_d - 1), w);
      if (kind == 0) {
        if constexpr (J == 0) D.template logpdf_dk<K, A>(st, y, x, nullptr, o, w);
        else D.template logpdf_dy_dk<J, K, A>(st, y, x, nullptr, o, w);
      } else if constexpr (J == 0 && K + A <= 4) {
        const int sa = kind == 3 ? ld::theta_slot : 0, sb = kind == 1 ? 0 : ld::theta_slot;
        D.template expected_info_dk<K, A>(st, sa, sb, nullptr, x, nullptr, o, w);
      }
    } else if constexpr (J == 0 && K == 0) {
      const ld::multinomial_distrib M(d, par(i, 2 * max_d));
      const auto st = M.theta_state(nullptr, w);
      if (kind == 0) M.template logpmf_dk<0, A>(st, y, x, nullptr, o, w);
      else if constexpr (A <= 4) M.template expected_info_dk<0, A>(st, 0, 0, nullptr, x, nullptr, o, w);
    }
  }
  template <int J, int K>
  KOKKOS_INLINE_FUNCTION void run_a(int i, int A) const {
    if constexpr (J + K <= 6) {
      switch (A) {
        case 0: run<J, K, 0>(i); break;
        case 1: if constexpr (J + K + 1 <= 6) run<J, K, 1>(i); break;
        case 2: if constexpr (J + K + 2 <= 6) run<J, K, 2>(i); break;
        case 3: if constexpr (J + K + 3 <= 6) run<J, K, 3>(i); break;
        case 4: if constexpr (J + K + 4 <= 6) run<J, K, 4>(i); break;
        case 5: if constexpr (J + K + 5 <= 6) run<J, K, 5>(i); break;
        default: if constexpr (J + K + 6 <= 6) run<J, K, 6>(i); break;
      }
    }
  }
  template <int J>
  KOKKOS_INLINE_FUNCTION void run_k(int i, int K, int A) const {
    switch (K) {
      case 0: run_a<J, 0>(i, A); break;
      case 1: run_a<J, 1>(i, A); break;
      case 2: run_a<J, 2>(i, A); break;
      case 3: run_a<J, 3>(i, A); break;
      case 4: run_a<J, 4>(i, A); break;
      case 5: run_a<J, 5>(i, A); break;
      default: run_a<J, 6>(i, A); break;
    }
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int J = meta(i, 3), A = meta(i, 4), K = meta(i, 5);
    switch (J) {
      case 0: run_k<0>(i, K, A); break;
      case 1: run_k<1>(i, K, A); break;
      case 2: run_k<2>(i, K, A); break;
      case 3: run_k<3>(i, K, A); break;
      default: run_k<4>(i, K, A); break;
    }
  }
};

// the values of every block, from the device
std::vector<std::vector<double>> evaluate(const std::vector<Block>& blocks) {
  const int n = int(blocks.size());
  const int ws = std::max(ld::dirichlet_distrib(max_d).scratch_size(), ld::multinomial_distrib(max_d, 1.0).scratch_size());
  Eval e{Kokkos::View<int* [6]>("meta", n), Kokkos::View<double* [2 * max_d + 1], Kokkos::LayoutRight>("par", n),
         Kokkos::View<double* [max_out], Kokkos::LayoutRight>("out", n), Kokkos::View<double**, Kokkos::LayoutRight>("work", n, ws)};
  auto meta = Kokkos::create_mirror_view(e.meta);
  auto par = Kokkos::create_mirror_view(e.par);
  for (int i = 0; i < n; ++i) {
    const Block& b = blocks[i];
    const int m[6] = {b.law, b.d, b.kind, b.J, b.A, b.K};
    for (int q = 0; q < 6; ++q) meta(i, q) = m[q];
    for (int q = 0; q < 2 * max_d + 1; ++q) par(i, q) = 0.0;
    for (int q = 0; q + 1 < b.d; ++q) par(i, q) = b.x[q];
    par(i, max_d - 1) = b.phi;
    for (int q = 0; q < b.d; ++q) par(i, max_d + q) = b.y[q];
    par(i, 2 * max_d) = b.n;
  }
  Kokkos::deep_copy(e.meta, meta);
  Kokkos::deep_copy(e.par, par);
  Kokkos::parallel_for("simplex_blocks", Kokkos::RangePolicy<>(0, n), e);
  Kokkos::fence();
  auto out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e.out);
  std::vector<std::vector<double>> v(n);
  for (int i = 0; i < n; ++i)
    for (int p : blocks[i].pos) {
      if (int(v[i].size()) <= p) v[i].resize(p + 1, 0.0);
      v[i][p] = out(i, p);
    }
  return v;
}

// worst ulp1N by [law - 1][kind][J > 0]
struct Worst {
  double v[2][4][2] = {};
};
Worst measure(const std::vector<Block>& blocks) {
  const auto got = evaluate(blocks);
  const double eps = std::ldexp(1.0, -52);
  Worst w;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const Block& b = blocks[i];
    double scale = 1.0, err = 0.0;
    for (double h : b.hi) scale = std::fmax(scale, std::fabs(h));
    for (std::size_t r = 0; r < b.pos.size(); ++r) {
      const double e = std::fabs((got[i][b.pos[r]] - b.hi[r]) - b.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    double& wv = w.v[b.law - 1][b.kind][b.J > 0];
    wv = std::fmax(wv, err / (scale * eps));
  }
  return w;
}

void check(const char* what, const Worst& w, const double (&bound)[2][4][2]) {
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  const char* names[2][4] = {{"dirichlet blocks", "dirichlet info (eta,eta)", "dirichlet info (eta,phi)",
                              "dirichlet info (phi,phi)"},
                             {"multinomial blocks", "multinomial info", "", ""}};
  for (int l = 0; l < 2; ++l)
    for (int k = 0; k < 4; ++k)
      for (int j = 0; j < 2; ++j) {
        if (!names[l][k][0] || (j == 1 && (l == 1 || k > 0))) continue;
        if (print) std::printf("%s %-26s %s: %9.3g ulp1N\n", what, names[l][k], j ? "y" : "-", w.v[l][k][j]);
        CHECK_MESSAGE(w.v[l][k][j] <= bound[l][k][j],
                      what << " " << names[l][k] << (j ? " (y)" : "") << ": " << w.v[l][k][j] << " ulp1N");
      }
}

void mpmath_case() {
  const auto blocks = read_blocks(std::string(LAPLACE_REFERENCE_DIR) + "/distrib_simplex.csv");
  REQUIRE(blocks.size() > 1000);
  // measured (gcc / CUDA): blocks 44.8 / 33.2, y 14.3 / 4.82, information 4.06 / 4.27,
  // 4.83 / 5.95, 3.06 / 11.1; multinomial 1.65 / 1.65, 0.717 / 0.717; the larger, half
  // again, at least 1
  const double bound[2][4][2] = {{{68, 22}, {6.5, 0}, {9, 0}, {17, 0}}, {{2.5, 0}, {1.1, 0}, {0, 0}, {0, 0}}};
  check("mpmath", measure(blocks), bound);
}

void statmodels7_case() {
  const auto blocks = read_blocks(std::string(LAPLACE_REFERENCE_DIR) + "/distrib_simplex_s7.csv");
  REQUIRE(blocks.size() > 100);
  // statmodels7's own accuracy, not Laplace's: its direct routes at phi = 300 (439 ulp1N in
  // the blocks) and its lgamma log-mass at size 1000 (628; Laplace's is 1.7 against mpmath);
  // measured the same on gcc and CUDA (439, 0, 2.26, 1.29, 0.281, 628, 1.07), half again
  const double bound[2][4][2] = {{{660, 1}, {3.4, 0}, {2, 0}, {1, 0}}, {{950, 0}, {1.6, 0}, {0, 0}, {0, 0}}};
  check("statmodels7", measure(blocks), bound);
}

// draws: n units at one parameter point, each from its own counter stream
struct Draw {
  int law, d;
  double n_trials;
  Kokkos::View<double[max_d]> x;
  double phi;
  Kokkos::View<double**, Kokkos::LayoutRight> y;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double th[1] = {phi};
    double w[1];
    if (law == 1) {
      const ld::dirichlet_distrib D(d);
      D.rng(D.theta_state(th, w), &x(0), 20261117u, std::uint64_t(i), &y(i, 0));
    } else {
      const ld::multinomial_distrib M(d, n_trials);
      M.rng(M.theta_state(th, w), &x(0), 20261117u, std::uint64_t(i), &y(i, 0));
    }
  }
};

std::vector<std::vector<double>> draw(int law, int d, const std::vector<double>& x, double phi, double n_trials, int n) {
  Kokkos::View<double[max_d]> xv("x");
  auto xh = Kokkos::create_mirror_view(xv);
  for (int q = 0; q + 1 < d; ++q) xh(q) = x[q];
  Kokkos::deep_copy(xv, xh);
  Kokkos::View<double**, Kokkos::LayoutRight> y("y", n, d);
  Kokkos::parallel_for("simplex_draws", Kokkos::RangePolicy<>(0, n), Draw{law, d, n_trials, xv, phi, y});
  Kokkos::fence();
  auto yh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  std::vector<std::vector<double>> r(n, std::vector<double>(d));
  for (int i = 0; i < n; ++i)
    for (int q = 0; q < d; ++q) r[i][q] = yh(i, q);
  return r;
}

void draws_case() {
  const std::vector<double> x = {0.3, -0.2};
  const int n = 40000, d = 3;
  for (int law = 1; law <= 2; ++law) {
    const double phi = 8.0, N = 20.0;
    const auto y = draw(law, d, x, phi, N, n);
    std::vector<double> th = {phi}, mean(d), cov(6), w(d);
    if (law == 1) {
      const ld::dirichlet_distrib D(d);
      D.mean(th.data(), x.data(), mean.data());
      D.covariance(th.data(), x.data(), cov.data(), w.data());
    } else {
      const ld::multinomial_distrib M(d, N);
      M.mean(th.data(), x.data(), mean.data());
      M.covariance(th.data(), x.data(), cov.data(), w.data());
    }
    std::vector<double> m(d, 0.0), c(6, 0.0);
    for (const auto& r : y) {
      double s = 0.0;
      for (int q = 0; q < d; ++q) {
        m[q] += r[q] / n;
        s += r[q];
        if (law == 2) CHECK(r[q] == std::floor(r[q]));
      }
      CHECK(std::fabs(s - (law == 1 ? 1.0 : N)) <= 1e-12 * (law == 1 ? 1.0 : N));
    }
    for (const auto& r : y)
      for (int a = 0; a < d; ++a)
        for (int b = 0; b <= a; ++b) c[a * (a + 1) / 2 + b] += (r[a] - m[a]) * (r[b] - m[b]) / n;
    for (int q = 0; q < d; ++q) {
      const double se = std::sqrt(cov[q * (q + 1) / 2 + q] / n);
      CHECK_MESSAGE(std::fabs(m[q] - mean[q]) <= 4.5 * se, "law " << law << " mean " << q);
    }
    for (int k = 0; k < 6; ++k) CHECK_MESSAGE(std::fabs(c[k] - cov[k]) <= 0.04 * std::fabs(cov[0]), "law " << law << " cov " << k);
  }
  // a tiny shape: every coordinate finite, the largest positive, the sum one
  const auto y = draw(1, d, x, 0.01, 0.0, 2000);
  for (const auto& r : y) {
    double s = 0.0, mx = 0.0;
    for (double v : r) {
      CHECK(std::isfinite(v));
      CHECK(v >= 0.0);
      s += v;
      mx = std::fmax(mx, v);
    }
    CHECK(mx > 0.0);
    CHECK(std::fabs(s - 1.0) <= 1e-12);
  }
}

void start_case() {
  // the start value recovers eta and, for the Dirichlet, phi from the moments
  const std::vector<double> x = {0.3, -0.2};
  const auto y = draw(1, 3, x, 8.0, 0.0, 40000);
  std::vector<double> flat;
  for (const auto& r : y) flat.insert(flat.end(), r.begin(), r.end());
  const auto s = ld::summarize_mv_sample(flat.data(), int(y.size()), 3);
  const ld::dirichlet_distrib D(3);
  double th[1], xs[2];
  D.start_value(s, th, xs);
  CHECK(std::fabs(xs[0] - x[0]) < 0.02);
  CHECK(std::fabs(xs[1] - x[1]) < 0.02);
  CHECK(std::fabs(th[0] - 8.0) < 0.4);
  const auto z = draw(2, 3, x, 0.0, 20.0, 40000);
  std::vector<double> flat2;
  for (const auto& r : z) flat2.insert(flat2.end(), r.begin(), r.end());
  const auto s2 = ld::summarize_mv_sample(flat2.data(), int(z.size()), 3);
  const ld::multinomial_distrib M(3, 20.0);
  M.start_value(s2, nullptr, xs);
  CHECK(std::fabs(xs[0] - x[0]) < 0.02);
  CHECK(std::fabs(xs[1] - x[1]) < 0.02);
}

}  // namespace simplex_test

TEST_CASE("dirichlet and multinomial against mpmath") { simplex_test::mpmath_case(); }
TEST_CASE("dirichlet and multinomial against statmodels7") { simplex_test::statmodels7_case(); }
TEST_CASE("dirichlet and multinomial draws") { simplex_test::draws_case(); }
TEST_CASE("dirichlet and multinomial start values") { simplex_test::start_case(); }
