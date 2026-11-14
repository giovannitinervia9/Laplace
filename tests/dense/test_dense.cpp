// laplace::dense (include/laplace/dense/dense.hpp) on the matrices of
// tests/data/reference/dense.csv (tools/reference/dense.py): eigenvalues and
// singular values against mpmath, the identities of each decomposition, the
// ranks, and the failures reported in the status or thrown. Every error is in
// units of eps (dense_cases.hpp says against which norm); the bounds are the
// errors measured on the gcc and CUDA builds (benchmarks/dense.cpp) times
// about 1.5, rounded up.

#include <doctest/doctest.h>

#include <laplace/dense/dense.hpp>

#include "dense_cases.hpp"

#include <Kokkos_Core.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace {

namespace ld = laplace::dense;
using dense_cases::host_matrix;
using dense_cases::host_vector;

// Read in each test case: the Views must be freed before Kokkos::finalize.
std::vector<dense_cases::dense_case> cases() {
  return dense_cases::read_dense_cases(std::string(LAPLACE_REFERENCE_DIR) + "/dense.csv");
}

// A View in device memory is rejected at compile time.
#ifdef KOKKOS_ENABLE_CUDA
static_assert(!ld::detail::is_host_double_view<Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::CudaSpace>, 2>());
#endif
static_assert(ld::detail::is_host_double_view<host_matrix, 2>());

}  // namespace

TEST_CASE("dense: symmetric eigendecomposition, Cholesky and triangular solve") {
  int seen = 0;
  for (const auto& c : cases()) {
    if (c.kind != "symmetric") continue;
    ++seen;
    const auto e = dense_cases::measure_case(c);
    INFO("case " << c.number << ", n = " << c.n);
    CHECK(e.success);
    CHECK(e.values <= 19);  // measured 12.2 (gcc and CUDA)
    CHECK(e.residual <= 15);  // 9.85 / 9.92
    CHECK(e.orthonormality <= 110);  // 71.0 / 72.2 (n = 60)
    if (e.cholesky >= 0) {
      CHECK(e.cholesky <= 1);  // 0.56 / 0.62
      CHECK(e.triangular <= 1);  // 0.33 / 0.44
    }
  }
  CHECK(seen == 6);
}

TEST_CASE("dense: generalized symmetric eigendecomposition") {
  int seen = 0;
  for (const auto& c : cases()) {
    if (c.kind != "generalized") continue;
    ++seen;
    const auto e = dense_cases::measure_case(c);
    INFO("case " << c.number << ", n = " << c.n);
    CHECK(e.success);
    // values in units of eps kappa_i: 1270 at the zero eigenvalues of case 8
    // (cond(B) = 1e6), as the reduction with the Cholesky factor of B allows
    CHECK(e.values <= 1910);  // 1270.3 (gcc and CUDA)
    CHECK(e.residual <= 0.3);  // 0.16 / 0.19
    CHECK(e.orthonormality <= 0.15);  // 0.07 / 0.10
  }
  CHECK(seen == 3);
}

TEST_CASE("dense: SVD, pivoted QR, ranks and null spaces") {
  int seen = 0;
  for (const auto& c : cases()) {
    if (c.kind != "rectangular") continue;
    ++seen;
    const auto e = dense_cases::measure_case(c);
    INFO("case " << c.number << ", " << c.m << " x " << c.n);
    CHECK(e.success);
    CHECK(e.svd_rank == c.rank);
    CHECK(e.qr_rank == c.rank);
    CHECK(e.values <= 12);  // measured 7.93 (gcc and CUDA)
    CHECK(e.residual <= 22);  // 14.38 / 14.39
    CHECK(e.orthonormality <= 67);  // 44.17 / 44.53
    CHECK(e.qr_residual <= 2.4);  // 1.55 / 1.56
    CHECK(e.qr_orthonormality <= 22);  // 12.33 / 14.12
    if (c.m < c.n) {
      CHECK(e.null_columns == c.n - c.rank);
      CHECK(e.null_residual <= 2.7);  // 0.94 / 1.74
      CHECK(e.null_orthonormality <= 20);  // 12.09 / 13.07
    }
  }
  CHECK(seen == 8);
}

TEST_CASE("dense: null space of no constraints, explicit rank tolerance") {
  host_matrix C("C", 0, 4), T("T", 1, 1);
  auto st = ld::null_space_basis(C, T);
  CHECK(st.success);
  CHECK(st.rank == 0);
  REQUIRE(T.extent(0) == 4);
  REQUIRE(T.extent(1) == 4);
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) CHECK(T(i, j) == (i == j ? 1.0 : 0.0));

  // diag(1, 1e-6): rank 2 by default, rank 1 with relative_tolerance 1e-3
  host_matrix A("A", 2, 2), U("U", 2, 2), V("V", 2, 2);
  host_vector s("s", 2);
  A(0, 0) = 1.0;
  A(1, 1) = 1e-6;
  CHECK(ld::svd_decomposition(A, U, s, V).rank == 2);
  CHECK(ld::svd_decomposition(A, U, s, V, 1e-3).rank == 1);
  host_matrix Q("Q", 2, 2), R("R", 2, 2);
  Kokkos::View<int*, Kokkos::HostSpace> perm("perm", 2);
  CHECK(ld::pivoted_qr_decomposition(A, Q, R, perm).rank == 2);
  CHECK(ld::pivoted_qr_decomposition(A, Q, R, perm, 1e-3).rank == 1);
  CHECK(ld::null_space_basis(A, T, 1e-3).rank == 1);
  CHECK(T.extent(1) == 1);
}

TEST_CASE("dense: failures reported in the status") {
  const int n = 3;
  host_matrix A("A", n, n), B("B", n, n), L("L", n, n), V("V", n, n);
  host_vector w("w", n);
  // indefinite: diag(1, -1, 2)
  A(0, 0) = 1.0;
  A(1, 1) = -1.0;
  A(2, 2) = 2.0;
  for (int i = 0; i < n; ++i) B(i, i) = 1.0;

  auto st = ld::cholesky_decomposition(A, L);
  CHECK_FALSE(st.success);
  CHECK(st.reason == ld::dense_failure::not_positive_definite);

  st = ld::generalized_symmetric_eigen_decomposition(B, A, w, V);
  CHECK_FALSE(st.success);
  CHECK(st.reason == ld::dense_failure::not_positive_definite);

  // a NaN in the lower triangle is seen, one in the upper triangle is not read
  host_matrix N("N", n, n);
  Kokkos::deep_copy(N, B);
  N(0, 2) = std::numeric_limits<double>::quiet_NaN();
  CHECK(ld::cholesky_decomposition(N, L).success);
  N(2, 0) = std::numeric_limits<double>::quiet_NaN();
  st = ld::symmetric_eigen_decomposition(N, w, V);
  CHECK_FALSE(st.success);
  CHECK(st.reason == ld::dense_failure::non_finite_input);
  CHECK(ld::cholesky_decomposition(N, L).reason == ld::dense_failure::non_finite_input);

  // a zero on the diagonal of a triangular matrix
  host_matrix T("T", n, n), X("X", n, 1);
  T(0, 0) = 1.0;
  T(2, 2) = 1.0;
  st = ld::triangular_solve(T, X, ld::dense_triangle::lower, ld::dense_operation::identity);
  CHECK_FALSE(st.success);
  CHECK(st.reason == ld::dense_failure::singular);
}

TEST_CASE("dense: wrong dimensions throw") {
  host_matrix A("A", 3, 3), L("L", 2, 2), R("R", 3, 4), Q("Q", 3, 2), U("U", 3, 3), V("V", 4, 4);
  host_vector s("s", 3), w("w", 2);
  Kokkos::View<int*, Kokkos::HostSpace> perm("perm", 3);
  CHECK_THROWS_AS(ld::cholesky_decomposition(R, U), std::invalid_argument);  // not square
  CHECK_THROWS_AS(ld::cholesky_decomposition(A, L), std::invalid_argument);
  CHECK_THROWS_AS(ld::symmetric_eigen_decomposition(A, w, U), std::invalid_argument);
  CHECK_THROWS_AS(ld::svd_decomposition(A, U, s, V), std::invalid_argument);
  CHECK_THROWS_AS(ld::pivoted_qr_decomposition(A, Q, R, perm), std::invalid_argument);
  CHECK_THROWS_AS(ld::triangular_solve(A, V, ld::dense_triangle::lower, ld::dense_operation::identity),
                  std::invalid_argument);
}
