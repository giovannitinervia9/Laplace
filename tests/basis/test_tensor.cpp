// Tensor product bases (include/laplace/basis/tensor.hpp) against the
// references of tools/reference/basis_tensor.py, which multiply the margins'
// values computed from their definitions (exact rationals for the B-splines,
// integer coefficients for Legendre, mpmath for Fourier). Four cases: two
// B-splines; B-spline x cyclic B-spline (x_2 over several periods); Legendre x
// Fourier; B-spline x cyclic x Legendre. Rows and expansions of several
// multi-indices (integrals as k_j = -1) are evaluated in a Kokkos kernel on the
// default execution space; Gram matrices and the thin plate penalty on the
// host.
// Errors: rows in units of eps times the row's largest reference value;
// expansions in eps times sum_c |beta_c B_c| (both at least the smallest
// denormal); Gram and penalty entries in eps
// times the largest entry.
// Then: rows equal the products of the margins' own rows (also outside the
// domain), NaN propagation, the Kronecker transforms (T'G T = I and both
// penalty components diagonal with Demmler-Reinsch margins), their
// composition, marginal_operator_penalty_matrix, fill_tensor_basis_rows.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/basis/bspline.hpp>
#include <laplace/basis/cyclic_bspline.hpp>
#include <laplace/basis/fourier.hpp>
#include <laplace/basis/legendre.hpp>
#include <laplace/basis/tensor.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
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

namespace basis_tensor_test {

namespace lb = laplace::basis;
constexpr double eps = std::numeric_limits<double>::epsilon();

bool print_errors() {
  const char* e = std::getenv("LAPLACE_PRINT_ERRORS");
  return e && e[0] == '1';
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> f;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, c)) f.push_back(x);
  if (!s.empty() && s.back() == c) f.push_back("");
  return f;
}
double hex(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

struct case_data {
  std::vector<double> beta;
  std::vector<std::string> korders;                          // in file order
  std::map<std::string, std::vector<int>> korder;            // "a;b" -> k
  std::map<int, std::vector<double>> point;                  // p -> x
  std::map<std::pair<std::string, int>, std::map<int, double>> row;  // (k, p) -> column -> value
  std::map<std::pair<std::string, int>, double> expansion;
  std::map<std::pair<std::string, std::string>, std::map<int, double>> gram;
  std::map<int, double> thin_plate;
};

std::vector<int> ints(const std::string& s) {
  std::vector<int> v;
  for (auto& f : split(s, ';')) v.push_back(std::stoi(f));
  return v;
}

std::vector<case_data> read_cases() {
  std::vector<case_data> cs(4);
  std::ifstream rf(std::string(LAPLACE_REFERENCE_DIR) + "/basis_tensor.csv");
  REQUIRE(rf.good());
  std::string line;
  std::getline(rf, line);
  while (std::getline(rf, line)) {
    auto f = split(line, ',');
    auto& c = cs[std::stoi(f[0])];
    const int q = std::stoi(f[1]);
    const double v = hex(f[7]);
    if (q == 4) {
      c.beta.push_back(v);
      continue;
    }
    if (q == 0 || q == 1) {
      const int p = std::stoi(f[4]);
      if (!c.korder.count(f[2])) {
        c.korder[f[2]] = ints(f[2]);
        c.korders.push_back(f[2]);
      }
      if (!c.point.count(p)) {
        std::vector<double> x;
        for (auto& s : split(f[5], ';')) x.push_back(hex(s));
        c.point[p] = x;
      }
      if (q == 0) c.row[{f[2], p}][std::stoi(f[6])] = v;
      else c.expansion[{f[2], p}] = v;
    } else if (q == 2) {
      c.gram[{f[2], f[3]}][std::stoi(f[6])] = v;
    } else {
      c.thin_plate[std::stoi(f[6])] = v;
    }
  }
  return cs;
}

using DV = Kokkos::View<double*>;
using DV2 = Kokkos::View<double**, Kokkos::LayoutRight>;
using IV2 = Kokkos::View<int**, Kokkos::LayoutRight>;

// one (multi-index, point) pair per index: the row and the expansion
template <class TB>
struct eval_functor {
  static constexpr int D = TB::margin_count;
  TB b;
  int P;
  DV2 x;       // P x D
  IV2 k;       // NK x D
  DV beta;     // K
  DV2 out;     // NK P x K
  IV2 first;   // NK P x D
  DV expn;     // NK P
  DV2 scratch; // NK P x sum K_j
  KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
    const int kk = idx / P, p = idx % P;
    double xi[D];
    int ki[D], fi[D];
    for (int j = 0; j < D; ++j) {
      xi[j] = x(p, j);
      ki[j] = k(kk, j);
    }
    b.row_partial(xi, ki, fi, &out(idx, 0), &scratch(idx, 0));
    for (int j = 0; j < D; ++j) first(idx, j) = fi[j];
    expn(idx) = b.expansion_partial(xi, ki, beta.data(), &scratch(idx, 0));
  }
};

struct errors {
  double row = 0, expansion = 0, gram = 0, penalty = 0;
};

// eps times the scale, at least the smallest denormal: at a point one ulp above
// lower the integral rows are denormal
double ulp(double scale) { return std::max(eps * scale, std::numeric_limits<double>::denorm_min()); }

double worse(double a, double e) { return (e != e || a != a) ? std::numeric_limits<double>::quiet_NaN() : std::max(a, e); }

// device tensor, host tensor (same margins), reference data
template <class TB>
errors check_case(const TB& dev, const TB& host, const case_data& c, const char* name) {
  constexpr int D = TB::margin_count;
  const int K = host.dimension(), P = static_cast<int>(c.point.size()), NK = static_cast<int>(c.korders.size());
  REQUIRE(static_cast<int>(c.beta.size()) == K);
  int S = 0;
  {
    std::array<int, D> m1;
    m1.fill(-1);
    S = host.scratch_size(m1.data());  // the largest scratch: every margin integrated
  }
  DV2 x("x", P, D);
  IV2 k("k", NK, D);
  DV beta("beta", K);
  DV2 out("out", NK * P, K);
  IV2 first("first", NK * P, D);
  DV expn("expn", NK * P);
  DV2 scratch("scratch", NK * P, S);
  auto xh = Kokkos::create_mirror_view(x);
  auto kh = Kokkos::create_mirror_view(k);
  auto bh = Kokkos::create_mirror_view(beta);
  for (auto& [p, v] : c.point)
    for (int j = 0; j < D; ++j) xh(p, j) = v[j];
  for (int a = 0; a < NK; ++a)
    for (int j = 0; j < D; ++j) kh(a, j) = c.korder.at(c.korders[a])[j];
  for (int i = 0; i < K; ++i) bh(i) = c.beta[i];
  Kokkos::deep_copy(x, xh);
  Kokkos::deep_copy(k, kh);
  Kokkos::deep_copy(beta, bh);
  Kokkos::parallel_for("basis_tensor_test::eval", NK * P, eval_functor<TB>{dev, P, x, k, beta, out, first, expn, scratch});
  Kokkos::fence();
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  auto fh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), first);
  auto eh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expn);

  errors e;
  for (int a = 0; a < NK; ++a) {
    const auto& ks = c.korders[a];
    const auto& kv = c.korder.at(ks);
    for (int p = 0; p < P; ++p) {
      const int idx = a * P + p;
      std::vector<double> got(K, 0.0);
      int fi[D];
      for (int j = 0; j < D; ++j) fi[j] = fh(idx, j);
      const int W = host.row_width(kv.data());
      for (int i = 0; i < W; ++i) got[host.column(kv.data(), fi, i)] += oh(idx, i);
      static const std::map<int, double> empty;  // a row of zeros (an integral at lower) has no line in the file
      const auto rit = c.row.find({ks, p});
      const auto& ref = rit == c.row.end() ? empty : rit->second;
      double scale = 0, wsum = 0;
      for (auto& [col, v] : ref) {
        scale = std::max(scale, std::fabs(v));
        wsum += std::fabs(c.beta[col] * v);
      }
      double err = 0;
      for (int col = 0; col < K; ++col) {
        const auto it = ref.find(col);
        const double r = it == ref.end() ? 0.0 : it->second;
        err = worse(err, std::fabs(got[col] - r) / ulp(scale));
      }
      e.row = worse(e.row, err);
      const double ex = c.expansion.at({ks, p});
      e.expansion = worse(e.expansion, std::fabs(eh(idx) - ex) / ulp(wsum));
    }
  }
  std::vector<double> g(static_cast<size_t>(K) * K);
  for (auto& [kl, ref] : c.gram) {
    const auto kv = ints(kl.first), lv = ints(kl.second);
    host.cross_gram(kv.data(), lv.data(), g.data());
    double scale = 0;
    for (auto& [i, v] : ref) scale = std::max(scale, std::fabs(v));
    for (int r = 0; r < K; ++r)
      for (int cc = 0; cc < K; ++cc) {
        const auto it = ref.find(r * K + cc);
        const double v = it == ref.end() ? 0.0 : it->second;
        e.gram = worse(e.gram, std::fabs(g[r + static_cast<size_t>(cc) * K] - v) / (eps * scale));
      }
  }
  if (print_errors())
    std::printf("tensor %-28s rows %6.2f  expansions %6.2f  gram %6.2f\n", name, e.row, e.expansion, e.gram);
  return e;
}

// owners of the margins of the four cases (as in basis_tensor.py)
struct margins {
  lb::bspline_basis_owner b0 = lb::bspline_basis_owner::uniform_clamped(0, 1, 7, 3);
  lb::bspline_basis_owner b1 = lb::bspline_basis_owner::uniform_clamped(0, 2, 5, 2);
  lb::bspline_basis_owner b2 = lb::bspline_basis_owner::uniform_clamped(0, 1, 6, 3);
  lb::cyclic_bspline_basis_owner c2 = lb::cyclic_bspline_basis_owner::uniform(0, 1, 5, 2);
  lb::legendre_basis l3 = lb::make_legendre_basis(0, 2, 3);
  lb::fourier_basis f3 = lb::make_fourier_basis(0, 1, 2);
  lb::bspline_basis_owner b4 = lb::bspline_basis_owner::uniform_clamped(0, 1, 5, 2);
  lb::cyclic_bspline_basis_owner c4 = lb::cyclic_bspline_basis_owner::uniform(0, 1, 6, 3);
  lb::legendre_basis l4 = lb::make_legendre_basis(-1, 1, 2);
};

// the row of the product equals the product of the margins' rows, on the host
template <class TB>
double product_identity(const TB& b, const double* x, const int* k) {
  constexpr int D = TB::margin_count;
  const int W = b.row_width(k);
  std::vector<double> out(W), scratch(b.scratch_size(k)), naive(W, 1.0);
  int first[D];
  b.row_partial(x, k, first, out.data(), scratch.data());
  // naive: margin by margin, the last fastest
  std::vector<std::vector<double>> r(D);
  lb::detail_tensor::host_each(b, [&](auto I, const auto& m) {
    r[I].resize(lb::detail_tensor::margin_width(m, k[I]));
    const int f = lb::detail_tensor::margin_row(m, k[I], x[I], r[I].data());
    CHECK(f == first[I]);
  });
  for (int flat = 0; flat < W; ++flat) {
    double p = 1.0;  // margin 1 first, as the in-place product
    int q = flat;
    std::vector<int> i(D);
    for (int j = D - 1; j >= 0; --j) {
      const int w = static_cast<int>(r[j].size());
      i[j] = q % w;
      q /= w;
    }
    for (int j = 0; j < D; ++j) p *= r[j][i[j]];
    naive[flat] = p;
  }
  double worst = 0;
  for (int i = 0; i < W; ++i) {
    if (std::isnan(naive[i]) != std::isnan(out[i])) return std::numeric_limits<double>::infinity();
    if (!std::isnan(naive[i])) worst = std::max(worst, std::fabs(out[i] - naive[i]));
  }
  return worst;
}

}  // namespace basis_tensor_test

using namespace basis_tensor_test;

TEST_CASE("tensor product bases: rows, expansions and Gram matrices against references") {
  static_assert(lb::multivariate_basis<lb::tensor_product_basis<lb::bspline_basis, lb::fourier_basis>>);
  static_assert(lb::multivariate_basis<
                lb::tensor_product_basis<lb::bspline_basis, lb::cyclic_bspline_basis, lb::legendre_basis>>);
  const auto cs = read_cases();
  margins m;
  // bounds: 1.5 times the worst measured over gcc and CUDA, per case
  {
    const lb::tensor_product_basis dev(m.b0.device(), m.b1.device());
    const lb::tensor_product_basis host(m.b0.host(), m.b1.host());
    const errors bound{3.5, 1.8, 2.6, 1.3};
    const auto e = check_case(dev, host, cs[0], "bspline x bspline");
    CHECK(e.row <= bound.row);
    CHECK(e.expansion <= bound.expansion);
    CHECK(e.gram <= bound.gram);
    // thin plate: three operators, d_xx, sqrt(2) d_xy, d_yy
    const int K = host.dimension();
    std::vector<double> s(static_cast<size_t>(K) * K, 0.0), t(s.size());
    for (auto& term : std::vector<lb::tensor_operator_term>{{1.0, {2, 0}}, {std::sqrt(2.0), {1, 1}}, {1.0, {0, 2}}}) {
      lb::tensor_operator_penalty_matrix(host, {term}, t.data());
      for (size_t i = 0; i < s.size(); ++i) s[i] += t[i];
    }
    double scale = 0, err = 0;
    for (auto& [i, v] : cs[0].thin_plate) scale = std::max(scale, std::fabs(v));
    for (int r = 0; r < K; ++r)
      for (int cc = 0; cc < K; ++cc) {
        const auto it = cs[0].thin_plate.find(r * K + cc);
        const double v = it == cs[0].thin_plate.end() ? 0.0 : it->second;
        err = worse(err, std::fabs(s[r + static_cast<size_t>(cc) * K] - v) / (eps * scale));
      }
    if (print_errors()) std::printf("tensor thin plate penalty         %6.2f\n", err);
    CHECK(err <= bound.penalty);
  }
  {
    const lb::tensor_product_basis dev(m.b2.device(), m.c2.device());
    const lb::tensor_product_basis host(m.b2.host(), m.c2.host());
    const errors bound{16.5, 4.5, 4.9, 0};
    const auto e = check_case(dev, host, cs[1], "bspline x cyclic");
    CHECK(e.row <= bound.row);
    CHECK(e.expansion <= bound.expansion);
    CHECK(e.gram <= bound.gram);
  }
  {
    lb::tensor_product_basis tb(m.l3, m.f3);
    const errors bound{3.2, 2.6, 1.9, 0};
    const auto e = check_case(tb, tb, cs[2], "legendre x fourier");
    CHECK(e.row <= bound.row);
    CHECK(e.expansion <= bound.expansion);
    CHECK(e.gram <= bound.gram);
  }
  {
    const lb::tensor_product_basis dev(m.b4.device(), m.c4.device(), m.l4);
    const lb::tensor_product_basis host(m.b4.host(), m.c4.host(), m.l4);
    const errors bound{6.0, 2.1, 4.0, 0};
    const auto e = check_case(dev, host, cs[3], "bspline x cyclic x legendre");
    CHECK(e.row <= bound.row);
    CHECK(e.expansion <= bound.expansion);
    CHECK(e.gram <= bound.gram);
  }
}

TEST_CASE("tensor product bases: rows are products of the margins' rows, also outside the domain") {
  margins m;
  lb::tensor_product_basis b(m.b0.host(), m.c2.host(), m.f3);
  const double xs[][3] = {{0.3, 0.7, 0.1}, {-0.4, 2.3, 0.5}, {1.7, -1.2, 3.4}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
  const int ks[][3] = {{0, 0, 0}, {1, 2, 3}, {-1, 0, 1}, {2, -1, -1}, {-1, -1, -1}, {6, 6, 6}};
  for (auto& x : xs)
    for (auto& k : ks) CHECK(product_identity(b, x, k) == 0.0);
  // a NaN coordinate makes the whole row NaN
  const double xn[3] = {0.5, std::numeric_limits<double>::quiet_NaN(), 0.5};
  for (auto& k : ks) {
    const int W = b.row_width(k);
    std::vector<double> out(W), scratch(b.scratch_size(k));
    int first[3];
    b.row_partial(xn, k, first, out.data(), scratch.data());
    for (double v : out) CHECK(std::isnan(v));
  }
  // the nan rule of a margin outside its domain
  auto bn = lb::bspline_basis_owner::uniform_clamped(0, 1, 7, 3, lb::outside_domain_rule::nan);
  lb::tensor_product_basis b2(bn.host(), m.l3);
  const double xo[2] = {1.5, 0.5};
  for (int k0 : {0, 1, -1}) {
    const int k[2] = {k0, 0};
    std::vector<double> out(b2.row_width(k)), scratch(b2.scratch_size(k));
    int first[2];
    b2.row_partial(xo, k, first, out.data(), scratch.data());
    for (double v : out) CHECK(std::isnan(v));
  }
  // expansion against the dot product with the scattered row
  const int K = b.dimension();
  std::vector<double> beta(K);
  for (int i = 0; i < K; ++i) beta[i] = std::sin(1.0 + i);
  for (auto& x : xs)
    for (auto& k : ks) {
      const int W = b.row_width(k);
      std::vector<double> out(W), scratch(b.scratch_size(k));
      int first[3];
      b.row_partial(x, k, first, out.data(), scratch.data());
      double dot = 0, mag = 0;
      for (int i = 0; i < W; ++i) {
        dot += out[i] * beta[b.column(k, first, i)];
        mag += std::fabs(out[i] * beta[b.column(k, first, i)]);
      }
      CHECK(std::fabs(b.expansion_partial(x, k, beta.data(), scratch.data()) - dot) <= 8 * eps * mag);
    }
}

TEST_CASE("tensor product bases: Kronecker transforms and penalties") {
  margins m;
  lb::tensor_product_basis b(m.b0.host(), m.b1.host());
  const int K0 = 7, K1 = 5, K = K0 * K1;
  using HM = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  const HM none("none", 0, 0);
  auto margin_transform = [&](const auto& mb, int Kj) {
    HM G("G", Kj, Kj), S("S", Kj, Kj);
    mb.cross_gram(0, 0, G.data());
    lb::operator_penalty_matrix(mb, lb::derivative_operator(2), S.data());
    return lb::make_basis_transform(Kj, none, G, S, 2);
  };
  const auto t0 = margin_transform(b.margin<0>(), K0), t1 = margin_transform(b.margin<1>(), K1);
  REQUIRE(t0.columns() == K0);
  REQUIRE(t1.columns() == K1);
  const auto t = lb::kronecker_transform({t0, t1});
  REQUIRE(t.rows() == K);
  REQUIRE(t.columns() == K);
  std::vector<double> g(static_cast<size_t>(K) * K), s0(g.size()), s1(g.size()), c(g.size());
  const int zero[2] = {0, 0};
  b.cross_gram(zero, zero, g.data());
  lb::marginal_operator_penalty_matrix(b, 0, lb::derivative_operator(2), s0.data());
  lb::marginal_operator_penalty_matrix(b, 1, lb::derivative_operator(2), s1.data());
  // marginal_operator_penalty_matrix with D^2 is the Gram of the multi-indices (2, 0)
  // and (0, 2)
  std::vector<double> g20(g.size()), g02(g.size());
  const int k20[2] = {2, 0}, k02[2] = {0, 2};
  b.cross_gram(k20, k20, g20.data());
  b.cross_gram(k02, k02, g02.data());
  CHECK(s0 == g20);
  CHECK(s1 == g02);
  // T' G T = I, T' S_j T diagonal
  t.congruence(g.data(), c.data());
  double ortho = 0;
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < K; ++j) ortho = std::max(ortho, std::fabs(c[i + j * K] - (i == j ? 1.0 : 0.0)));
  for (const auto* sj : {&s0, &s1}) {
    t.congruence(sj->data(), c.data());
    double off = 0, big = 0;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < K; ++j) {
        if (i == j) big = std::max(big, std::fabs(c[i + j * K]));
        else off = std::max(off, std::fabs(c[i + j * K]));
      }
    if (print_errors()) std::printf("tensor kron DR: off-diagonal / largest %.3g eps\n", off / big / eps);
    CHECK(off <= 64 * eps * big);
  }
  if (print_errors()) std::printf("tensor kron DR: max|T'GT - I| %.3g eps\n", ortho / eps);
  CHECK(ortho <= 256 * eps);
  // composition: a zero-integral constraint on the product after the margins
  HM C("C", 1, K);
  {
    const int ki[2] = {-1, -1};
    const double up[2] = {1.0, 2.0};
    int first[2];
    std::vector<double> scratch(b.scratch_size(ki));
    b.row_partial(up, ki, first, C.data(), scratch.data());  // the column totals; C is 1 x K
  }
  HM Cg("Cg", 1, K);
  for (int j = 0; j < K; ++j) {
    double v = 0;
    for (int i = 0; i < K; ++i) v += C(0, i) * t.matrix()(i, j);
    Cg(0, j) = v;
  }
  const auto tp = lb::make_basis_transform(K, Cg, none, none, 0);
  const auto tc = lb::compose_basis_transforms(t, tp);
  REQUIRE(tc.rows() == K);
  REQUIRE(tc.columns() == K - 1);
  double ct = 0, cmax = 0;
  for (int i = 0; i < K; ++i) cmax = std::max(cmax, std::fabs(C(0, i)));
  for (int j = 0; j < K - 1; ++j) {
    double v = 0;
    for (int i = 0; i < K; ++i) v += C(0, i) * tc.matrix()(i, j);
    ct = std::max(ct, std::fabs(v));
  }
  if (print_errors()) std::printf("tensor compose: max|C T| %.3g eps max|C|\n", ct / cmax / eps);
  CHECK(ct <= 256 * eps * cmax);
  CHECK_THROWS(lb::compose_basis_transforms(tp, t));
  const int bad[2] = {7, 0};
  CHECK_THROWS(b.cross_gram(bad, zero, g.data()));
  CHECK_THROWS(lb::tensor_operator_penalty_matrix(b, {{1.0, {0, 3}}}, g.data()));  // margin 1 has degree 2
}

namespace basis_tensor_test {
// fill_tensor_basis_rows on the default device against row_partial on the host
double fill_rows_check() {
  margins m;
  const lb::tensor_product_basis dev(m.b2.device(), m.c2.device());
  const lb::tensor_product_basis host(m.b2.host(), m.c2.host());
  const int n = 50;
  const std::array<int, 2> k{1, -1};
  Kokkos::View<double**, Kokkos::LayoutRight> x("x", n, 2);
  auto xh = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) {
    xh(i, 0) = (i + 0.5) / n;
    xh(i, 1) = -1.0 + 3.0 * i / n;
  }
  Kokkos::deep_copy(x, xh);
  const int W = host.row_width(k.data());
  Kokkos::View<int**, Kokkos::LayoutRight> first("first", n, 2);
  Kokkos::View<double**, Kokkos::LayoutRight> values("values", n, W);
  lb::fill_tensor_basis_rows(dev, k, x, first, values);
  auto fh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), first);
  auto vh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  double worst = 0;
  std::vector<double> out(W), scratch(host.scratch_size(k.data()));
  for (int i = 0; i < n; ++i) {
    const double xi[2] = {xh(i, 0), xh(i, 1)};
    int f[2];
    host.row_partial(xi, k.data(), f, out.data(), scratch.data());
    if (f[0] != fh(i, 0) || f[1] != fh(i, 1)) return std::numeric_limits<double>::infinity();
    for (int j = 0; j < W; ++j) worst = std::max(worst, std::fabs(out[j] - vh(i, j)) / (eps * std::max(1.0, std::fabs(out[j]))));
  }
  return worst;
}
}  // namespace basis_tensor_test

TEST_CASE("tensor product bases: fill_tensor_basis_rows") {
  const double e = fill_rows_check();
  if (print_errors()) std::printf("tensor fill rows: device against host %.3g eps\n", e);
  CHECK(e <= 16.0);
}
