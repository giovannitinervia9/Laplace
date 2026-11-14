// Accuracy and speed of laplace::dense (include/laplace/dense/dense.hpp).
// Accuracy: every case of tests/data/reference/dense.csv (tools/reference/dense.py),
// the errors of dense_cases.hpp in units of eps. Speed: ms per call on one
// thread for K x K matrices, K = 20, 50, 200, 1000 (or the sizes given on the
// command line): A = G G^T / K + I with G uniform on (-1, 1), a general G
// for the SVD and the QR, three random constraint rows for the null space.
//
// Usage: dense [K ...]

#include <laplace/dense/dense.hpp>

#include "dense_cases.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace ld = laplace::dense;
using dense_cases::host_matrix;
using dense_cases::host_vector;

host_matrix uniform_matrix(int m, int n, std::uint64_t seed) {
  host_matrix a("G", m, n);
  std::uint64_t x = seed;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      x = x * 6364136223846793005ULL + 1442695040888963407ULL;
      a(i, j) = 2.0 * static_cast<double>(x >> 11) * 0x1p-53 - 1.0;
    }
  return a;
}

host_matrix positive_definite(int k, std::uint64_t seed) {
  const host_matrix g = uniform_matrix(k, k, seed);
  host_matrix a = dense_cases::product(g, false, g, true);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < k; ++i) a(i, j) = a(i, j) / k + (i == j ? 1.0 : 0.0);
  return a;
}

// ms per call: repeats until 0.2 s have passed (at least 3 calls)
template <class F>
double time_ms(F&& f) {
  using clock = std::chrono::steady_clock;
  f();  // warm-up
  int calls = 0;
  const auto start = clock::now();
  double elapsed = 0;
  while (calls < 3 || elapsed < 0.2) {
    f();
    ++calls;
    elapsed = std::chrono::duration<double>(clock::now() - start).count();
  }
  return 1e3 * elapsed / calls;
}

void accuracy() {
  const auto cases = dense_cases::read_dense_cases(std::string(LAPLACE_REFERENCE_DIR) + "/dense.csv");
  std::printf("Accuracy, units of eps (- : not applicable)\n");
  std::printf("%4s %-11s %9s | %8s %8s %8s | %8s %8s | %8s %8s | %8s %8s | %s\n", "case", "kind", "shape",
              "values", "resid", "orth", "chol", "trsv", "qr res", "qr orth", "null CT", "null TT",
              "ranks (expected svd qr cols)");
  auto cell = [](double v) {
    static char buf[4][16];
    static int k = 0;
    k = (k + 1) % 4;
    if (v < 0) std::snprintf(buf[k], 16, "%8s", "-");
    else std::snprintf(buf[k], 16, "%8.2f", v);
    return buf[k];
  };
  for (const auto& c : cases) {
    const auto e = dense_cases::measure_case(c);
    char shape[32];
    std::snprintf(shape, sizeof shape, "%dx%d", c.m, c.n);
    std::printf("%4d %-11s %9s | %s %s %s |", c.number, c.kind.c_str(), shape, cell(e.values), cell(e.residual),
                cell(e.orthonormality));
    std::printf(" %s %s |", cell(e.cholesky), cell(e.triangular));
    std::printf(" %s %s |", cell(e.qr_residual), cell(e.qr_orthonormality));
    std::printf(" %s %s |", cell(e.null_residual), cell(e.null_orthonormality));
    if (c.kind == "rectangular")
      std::printf(" %d %d %d %d", c.rank, e.svd_rank, e.qr_rank, e.null_columns);
    std::printf("%s\n", e.success ? "" : "  FAILED");
  }
}

void speed(const std::vector<int>& sizes) {
  std::printf("\nSpeed, ms per call, one thread\n");
  std::printf("%6s %10s %10s %10s %10s %10s %10s %10s\n", "K", "cholesky", "trsv K", "pivqr", "svd", "sym eig",
              "gen eig", "null 3xK");
  for (const int k : sizes) {
    const host_matrix a = positive_definite(k, 1), b = positive_definite(k, 2), g = uniform_matrix(k, k, 3);
    const host_matrix c = uniform_matrix(3, k, 4);
    host_matrix L("L", k, k), X("X", k, k), Q("Q", k, k), R("R", k, k), U("U", k, k), V("V", k, k);
    host_matrix T("T", 1, 1);
    host_vector s("s", k);
    Kokkos::View<int*, Kokkos::HostSpace> perm("perm", k);
    ld::cholesky_decomposition(a, L);
    const double t_chol = time_ms([&] { ld::cholesky_decomposition(a, L); });
    const double t_trsv = time_ms([&] {
      Kokkos::deep_copy(X, a);
      ld::triangular_solve(L, X, ld::dense_triangle::lower, ld::dense_operation::identity);
    });
    const double t_qr = time_ms([&] { ld::pivoted_qr_decomposition(g, Q, R, perm); });
    const double t_svd = time_ms([&] { ld::svd_decomposition(g, U, s, V); });
    const double t_eig = time_ms([&] { ld::symmetric_eigen_decomposition(a, s, V); });
    const double t_gen = time_ms([&] { ld::generalized_symmetric_eigen_decomposition(a, b, s, V); });
    const double t_null = time_ms([&] { ld::null_space_basis(c, T); });
    std::printf("%6d %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g\n", k, t_chol, t_trsv, t_qr, t_svd, t_eig,
                t_gen, t_null);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i)
    if (argv[i][0] != '-') sizes.push_back(std::atoi(argv[i]));
  if (sizes.empty()) sizes = {20, 50, 200, 1000};
  accuracy();
  speed(sizes);
  return 0;
}
