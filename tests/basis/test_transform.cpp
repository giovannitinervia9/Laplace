// Basis transforms (include/laplace/basis/transform.hpp).
// Against the mpmath references of tools/reference/basis_transform.py: cubic
// B-splines with uniform clamped knots on [0, 1], K = 10 and 40, penalty D^2
// and D^3, Gram G_00, with and without the zero-integral constraint. Errors in
// eps = 2^-52: eigenvalues in eps lambda_max, max|T'GT - I| in eps,
// off-diagonal of T'ST in eps lambda_max, max|C T| in eps max|C|.
// Then: an empirical Gram matrix, skipped steps, failures, coefficients and
// gradients on the default execution space, congruence, a given T.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/basis/bspline.hpp>
#include <laplace/basis/transform.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace basis_transform_test {

namespace lb = laplace::basis;
namespace ld = laplace::dense;
using M = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
constexpr double eps = 0x1.0p-52;

bool print_errors() {
  const char* p = std::getenv("LAPLACE_PRINT_ERRORS");
  return p && *p && *p != '0';
}

struct reference_case {
  int K = 0, m = 0, constrained = 0;
  std::vector<double> lambda;
};

std::map<int, reference_case> read_reference() {
  std::ifstream f(std::string(LAPLACE_REFERENCE_DIR) + "/basis_transform.csv");
  REQUIRE(f.good());
  std::map<int, reference_case> out;
  std::string line;
  std::getline(f, line);
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string v[6];
    for (auto& x : v) std::getline(ss, x, ',');
    reference_case& c = out[std::stoi(v[0])];
    c.K = std::stoi(v[1]);
    c.m = std::stoi(v[2]);
    c.constrained = std::stoi(v[3]);
    c.lambda.push_back(std::strtod(v[5].c_str(), nullptr));
  }
  return out;
}

M gram(const lb::bspline_basis& b, int k, int l) {
  M g("G", b.dimension(), b.dimension());
  b.cross_gram(k, l, g.data());
  return g;
}

M zero_integral(const lb::bspline_basis& b) {
  M c("C", 1, b.dimension());
  for (int j = 0; j < b.dimension(); ++j) c(0, j) = b.integral_column_total(j);
  return c;
}

// G = X'X / n on n points of a fixed LCG, uniform on [0, top]
M empirical_gram(const lb::bspline_basis& b, int n, double top) {
  const int K = b.dimension(), d = b.degree;
  M g("G", K, K);
  unsigned long long st = 12345;
  double row[lb::bspline_max_degree + 1];
  for (int i = 0; i < n; ++i) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    const double x = top * ((st >> 11) * 0x1.0p-53);
    const int f = b.row(x, row);
    for (int p = 0; p <= d; ++p)
      for (int q = 0; q <= d; ++q) g(f + p, f + q) += row[p] * row[q] / n;
  }
  return g;
}

// a' b c
M triple(const M& a, const M& b, const M& c) {
  M o("o", a.extent(1), c.extent(1));
  for (size_t i = 0; i < a.extent(1); ++i)
    for (size_t j = 0; j < c.extent(1); ++j) {
      long double s = 0;
      for (size_t k = 0; k < b.extent(0); ++k)
        for (size_t l = 0; l < b.extent(1); ++l) s += (long double)a(k, i) * b(k, l) * c(l, j);
      o(i, j) = (double)s;
    }
  return o;
}

M identity(int n) {
  M o("I", n, n);
  for (int i = 0; i < n; ++i) o(i, i) = 1;
  return o;
}

double orth_error(const M& T, const M& G) {  // max |T'GT - I|
  const M p = triple(T, G, T);
  double e = 0;
  for (size_t i = 0; i < p.extent(0); ++i)
    for (size_t j = 0; j < p.extent(1); ++j) e = std::fmax(e, std::fabs(p(i, j) - (i == j)));
  return e;
}

double offdiag(const M& T, const M& S) {  // max off-diagonal |T'ST|
  const M p = triple(T, S, T);
  double e = 0;
  for (size_t i = 0; i < p.extent(0); ++i)
    for (size_t j = 0; j < p.extent(1); ++j)
      if (i != j) e = std::fmax(e, std::fabs(p(i, j)));
  return e;
}

double constraint_error(const M& C, const M& T) {  // max |C T| / max |C|
  double e = 0, cm = 0;
  for (size_t j = 0; j < C.extent(1); ++j) cm = std::fmax(cm, std::fabs(C(0, j)));
  for (size_t j = 0; j < T.extent(1); ++j) {
    long double s = 0;
    for (size_t i = 0; i < T.extent(0); ++i) s += (long double)C(0, i) * T(i, j);
    e = std::fmax(e, std::fabs((double)s));
  }
  return e / cm;
}

}  // namespace basis_transform_test

using namespace basis_transform_test;

TEST_CASE("basis_transform: Demmler-Reinsch against mpmath") {
  // bounds 1.5 times the larger of the gcc and CUDA errors (54.7, 45.0, 6.7, 75.1)
  const double bound_lambda = 83, bound_orth = 68, bound_offd = 10, bound_cons = 113;
  const auto ref = read_reference();
  REQUIRE(ref.size() == 8);
  double worst[4] = {0, 0, 0, 0};
  for (const auto& [id, rc] : ref) {
    auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, rc.K, 3);
    const auto b = owner.host();
    const M G = gram(b, 0, 0), S = gram(b, rc.m, rc.m), C = rc.constrained ? zero_integral(b) : M();
    const int r = rc.m - rc.constrained, q = rc.K - rc.constrained;
    const auto bt = lb::make_basis_transform(rc.K, C, G, S, r);
    REQUIRE(bt.status().success);
    REQUIRE(bt.columns() == q);
    REQUIRE(bt.rows() == rc.K);
    const auto& w = bt.penalty_eigenvalues();
    REQUIRE(static_cast<int>(w.size()) == q);
    REQUIRE(static_cast<int>(rc.lambda.size()) == q);
    const double lmax = rc.lambda.back();
    double e[4] = {0, 0, 0, 0};
    for (int j = 0; j < q; ++j) {
      if (j < r) CHECK(w[j] == 0.0);
      CHECK(rc.lambda[j] == (j < r ? 0.0 : rc.lambda[j]));  // the reference has exactly r zeros
      e[0] = std::fmax(e[0], std::fabs(w[j] - rc.lambda[j]) / lmax / eps);
    }
    CHECK(rc.lambda[r] > 0.0);
    e[1] = orth_error(bt.matrix(), G) / eps;
    e[2] = offdiag(bt.matrix(), S) / lmax / eps;
    if (rc.constrained) e[3] = constraint_error(C, bt.matrix()) / eps;
    for (int k = 0; k < 4; ++k) worst[k] = std::fmax(worst[k], e[k]);
    if (print_errors())
      std::printf("transform K=%d m=%d constrained=%d: lambda %.1f orth %.1f offdiag %.1f constraint %.1f (eps)\n",
                  rc.K, rc.m, rc.constrained, e[0], e[1], e[2], e[3]);
    CHECK(e[0] <= bound_lambda);
    CHECK(e[1] <= bound_orth);
    CHECK(e[2] <= bound_offd);
    CHECK(e[3] <= bound_cons);
  }
  if (print_errors())
    std::printf("transform worst: lambda %.1f orth %.1f offdiag %.1f constraint %.1f (eps)\n", worst[0], worst[1],
                worst[2], worst[3]);
}

TEST_CASE("basis_transform: empirical Gram matrix") {
  auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 40, 3);
  const auto b = owner.host();
  const M G = empirical_gram(b, 200, 1.0), S = gram(b, 2, 2), C = zero_integral(b);
  const auto bt = lb::make_basis_transform(40, C, G, S, 1);
  REQUIRE(bt.status().success);
  REQUIRE(bt.columns() == 39);
  const auto& w = bt.penalty_eigenvalues();
  CHECK(w[0] == 0.0);
  CHECK(w[1] > 0.0);
  CHECK(bt.status().null_eigenvalue_ratio < 4 * eps);
  if (print_errors())
    std::printf("transform empirical G: orth %.1f offdiag %.1f constraint %.1f (eps)\n",
                orth_error(bt.matrix(), G) / eps, offdiag(bt.matrix(), S) / w.back() / eps,
                constraint_error(C, bt.matrix()) / eps);
  CHECK(orth_error(bt.matrix(), G) <= 39 * eps);
  CHECK(offdiag(bt.matrix(), S) <= 2 * eps * w.back());
  CHECK(constraint_error(C, bt.matrix()) <= 83 * eps);
}

TEST_CASE("basis_transform: skipped steps") {
  auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 12, 3);
  const auto b = owner.host();
  const M G = gram(b, 0, 0), S = gram(b, 2, 2), C = zero_integral(b), I = identity(12);

  const auto none = lb::make_basis_transform(12, M(), M(), M(), 0);
  REQUIRE(none.status().success);
  CHECK(none.columns() == 12);
  CHECK(orth_error(none.matrix(), I) == 0.0);
  CHECK(none.penalty_eigenvalues().empty());

  const auto c_only = lb::make_basis_transform(12, C, M(), M(), 0);
  REQUIRE(c_only.status().success);
  CHECK(c_only.columns() == 11);
  CHECK(orth_error(c_only.matrix(), I) <= 16 * eps);  // orthonormal columns
  CHECK(constraint_error(C, c_only.matrix()) <= 16 * eps);

  const auto g_only = lb::make_basis_transform(12, M(), G, M(), 0);
  REQUIRE(g_only.status().success);
  CHECK(g_only.columns() == 12);
  CHECK(orth_error(g_only.matrix(), G) <= 16 * eps);

  const auto s_only = lb::make_basis_transform(12, M(), M(), S, 2);  // Euclidean coordinates
  REQUIRE(s_only.status().success);
  const auto& w = s_only.penalty_eigenvalues();
  CHECK(w[0] == 0.0);
  CHECK(w[1] == 0.0);
  CHECK(w[2] > 0.0);
  CHECK(orth_error(s_only.matrix(), I) <= 16 * eps);
  CHECK(offdiag(s_only.matrix(), S) <= 8 * eps * w.back());
}

TEST_CASE("basis_transform: failures") {
  auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 20, 3);
  const auto b = owner.host();
  const M G = gram(b, 0, 0), S = gram(b, 2, 2), C = zero_integral(b);

  // data on [0, 0.5] only: the columns supported on (0.5, 1] are zero in G
  const M Gs = empirical_gram(b, 400, 0.5);
  const auto singular = lb::make_basis_transform(20, C, Gs, S, 1);
  CHECK_FALSE(singular.status().success);
  CHECK(singular.status().failed_step == lb::basis_transform_step::orthonormalization);
  CHECK(singular.status().dense_reason == ld::dense_failure::not_positive_definite);
  CHECK(singular.status().gram_rank > 0);
  CHECK(singular.status().gram_rank < 19);
  CHECK(singular.columns() == 0);

  // three null eigenvalues declared where D^2 has two
  const auto wrong_r = lb::make_basis_transform(20, M(), G, S, 3);
  CHECK_FALSE(wrong_r.status().success);
  CHECK(wrong_r.status().failed_step == lb::basis_transform_step::demmler_reinsch);
  CHECK(wrong_r.status().dense_reason == ld::dense_failure::none);
  CHECK(wrong_r.status().null_eigenvalue_ratio > 1e-8);
  CHECK(wrong_r.columns() == 0);

  CHECK_THROWS_AS(lb::make_basis_transform(0, M(), M(), M(), 0), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_basis_transform(21, C, M(), M(), 0), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_basis_transform(21, M(), G, M(), 0), std::invalid_argument);
  CHECK_THROWS_AS(lb::make_basis_transform(20, C, G, S, 20), std::invalid_argument);  // r > q = 19
  CHECK_THROWS_AS(lb::make_basis_transform(20, C, G, S, -1), std::invalid_argument);
}

TEST_CASE("basis_transform: coefficients, gradients, congruence") {
  const int K = 40;
  auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, K, 3);
  const auto b = owner.host();
  const M G = gram(b, 0, 0), S = gram(b, 2, 2), C = zero_integral(b);
  const auto bt = lb::make_basis_transform(K, C, G, S, 1);
  REQUIRE(bt.status().success);
  const int q = bt.columns();
  const M& T = bt.matrix();

  using DV = Kokkos::View<double*>;
  DV gamma("gamma", q), beta("beta", K), g_beta("g_beta", K), g_gamma("g_gamma", q);
  auto gamma_h = Kokkos::create_mirror_view(gamma);
  auto g_beta_h = Kokkos::create_mirror_view(g_beta);
  for (int j = 0; j < q; ++j) gamma_h(j) = std::sin(1.0 + j);
  for (int i = 0; i < K; ++i) g_beta_h(i) = std::cos(2.0 + 0.5 * i);
  Kokkos::deep_copy(gamma, gamma_h);
  Kokkos::deep_copy(g_beta, g_beta_h);

  // default execution space
  bt.expand_coefficients(gamma, beta);
  bt.reduce_gradient(g_beta, g_gamma);
  auto beta_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beta);
  auto g_gamma_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g_gamma);
  // host Views take the host route in every build
  Kokkos::View<double*, Kokkos::HostSpace> gamma_hv("gh", q), beta_hv("bh", K);
  Kokkos::deep_copy(gamma_hv, gamma_h);
  bt.expand_coefficients(gamma_hv, beta_hv);

  for (int i = 0; i < K; ++i) {
    long double s = 0, a = 0;
    for (int j = 0; j < q; ++j) {
      s += (long double)T(i, j) * gamma_h(j);
      a += std::fabs(T(i, j) * gamma_h(j));
    }
    CHECK(std::fabs(beta_h(i) - (double)s) <= 4 * q * eps * (double)a);
    CHECK(std::fabs(beta_hv(i) - (double)s) <= 4 * q * eps * (double)a);
  }
  for (int j = 0; j < q; ++j) {
    long double s = 0, a = 0;
    for (int i = 0; i < K; ++i) {
      s += (long double)T(i, j) * g_beta_h(i);
      a += std::fabs(T(i, j) * g_beta_h(i));
    }
    CHECK(std::fabs(g_gamma_h(j) - (double)s) <= 4 * K * eps * (double)a);
  }

  // the basis's own expansion with beta = T gamma equals sum_j gamma_j (T' b(x))_j
  std::vector<double> bh(K);
  for (int i = 0; i < K; ++i) bh[i] = beta_hv(i);
  for (double x : {0.0, 0.123, 0.5, 0.987, 1.0}) {
    double row[4];
    const int f = b.row(x, row);
    long double s = 0, a = 0;
    for (int j = 0; j < q; ++j)
      for (int p = 0; p < 4; ++p) {
        s += (long double)T(f + p, j) * row[p] * gamma_h(j);
        a += std::fabs(T(f + p, j) * row[p] * gamma_h(j));
      }
    CHECK(std::fabs(b.expansion(x, bh.data()) - (double)s) <= 8 * q * eps * (double)a);
  }

  // congruence: T'ST is the eigenvalue diagonal, T'GT the identity
  std::vector<double> ts(q * q), tg(q * q);
  bt.congruence(S.data(), ts.data());
  bt.congruence(G.data(), tg.data());
  const auto& w = bt.penalty_eigenvalues();
  for (int j = 0; j < q; ++j)
    for (int i = 0; i < q; ++i) {
      CHECK(std::fabs(ts[i + j * q] - (i == j ? w[j] : 0.0)) <= 64 * eps * w.back());
      CHECK(std::fabs(tg[i + j * q] - (i == j)) <= 68 * eps);
    }
}

TEST_CASE("basis_transform: a given T") {
  M T("T", 5, 3);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 3; ++j) T(i, j) = 0.25 * i - j;
  const auto bt = lb::basis_transform_from_matrix(T);
  CHECK(bt.status().success);
  CHECK(bt.rows() == 5);
  CHECK(bt.columns() == 3);
  CHECK(bt.penalty_eigenvalues().empty());
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 3; ++j) CHECK(bt.matrix()(i, j) == T(i, j));
  CHECK_THROWS_AS(lb::basis_transform_from_matrix(M()), std::invalid_argument);
}
