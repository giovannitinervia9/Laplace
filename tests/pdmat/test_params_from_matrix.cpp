// params_from_matrix of every structure (include/laplace/pdmat/*.hpp, helpers
// in detail/from_matrix.hpp), on the host:
//   - exact structures, and those whose fit recovers its own matrices:
//     params_from_matrix(matrix(theta)) == theta to a tolerance;
//   - the factor-analytic fit: the distance of
//     matrix(params_from_matrix(M)) from M;
//   - degenerate targets (a singular matrix, unit correlations, a zero
//     diagonal entry): finite theta, inside the domain.
// Set LAPLACE_PRINT_ERRORS=1 to print the errors.
#include <doctest/doctest.h>

#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/compound_symmetry.hpp>
#include <laplace/pdmat/diagonal.hpp>
#include <laplace/pdmat/factor_analytic.hpp>
#include <laplace/pdmat/reduced_rank.hpp>
#include <laplace/pdmat/toeplitz.hpp>
#include <laplace/pdmat/unit_diagonal_scaled.hpp>
#include <laplace/pdmat/unstructured.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
namespace bj = laplace::bijector;

static_assert(lp::weighted_sum_psdmat::constant_null_space && lp::weighted_sum_psdmat::supplies_null_space);
static_assert(!lp::reduced_rank_psdmat::constant_null_space && !lp::reduced_rank_psdmat::supplies_null_space);

bool print_errors() {
  const char* e = std::getenv("LAPLACE_PRINT_ERRORS");
  return e && e[0] == '1';
}

template <class S>
std::vector<double> matrix_of(const S& s, const std::vector<double>& theta) {
  std::vector<double> m(lp::packed_size(s.dim)), w(std::max(s.scratch_size(), 1));
  s.matrix(theta.data(), m.data(), w.data());
  return m;
}

template <class S>
std::vector<double> params_of(const S& s, const std::vector<double>& m) {
  std::vector<double> t(s.n_params());
  s.params_from_matrix(m.data(), t.data());
  return t;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) e = std::fmax(e, std::isfinite(a[k] - b[k]) ? std::fabs(a[k] - b[k]) : std::numeric_limits<double>::infinity());
  return e;
}

// max |M(params_from_matrix(M)) - M| / max |M|
template <class S>
double matrix_distance(const S& s, const std::vector<double>& m) {
  const auto back = matrix_of(s, params_of(s, m));
  double top = 0.0;
  for (double x : m) top = std::fmax(top, std::fabs(x));
  return max_abs_diff(back, m) / top;
}

// theta back from matrix(theta): the worst |difference|, checked against bound
template <class S>
void check_exact(const char* name, const S& s, const std::vector<double>& theta, double bound) {
  REQUIRE(int(theta.size()) == s.n_params());
  const double e = max_abs_diff(params_of(s, matrix_of(s, theta)), theta);
  CHECK_MESSAGE(e <= bound, name, ": max |theta - params_from_matrix(matrix(theta))| = ", e);
  if (print_errors()) std::printf("params_from_matrix %-40s max |theta error| %.2e\n", name, e);
}

// a positive definite p x p matrix (packed) with a spread of eigenvalues
std::vector<double> test_matrix(int p, unsigned seed) {
  std::vector<double> A(p * p), m(lp::packed_size(p));
  for (int k = 0; k < p * p; ++k) {
    seed = seed * 1103515245u + 12345u;
    A[k] = ((seed >> 8) % 2001) / 1000.0 - 1.0;
  }
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = i == j ? 0.5 : 0.0;
      for (int k = 0; k < p; ++k) s += A[i * p + k] * A[j * p + k];
      m[lp::packed_index(i, j)] = s;
    }
  return m;
}

std::vector<lp::weighted_sum_psdmat::host_matrix> penalty_matrices() {
  const int p = 6;
  std::vector<lp::weighted_sum_psdmat::host_matrix> S;
  for (int j = 0; j < 3; ++j) S.emplace_back("S", p, p);
  for (int i = 0; i + 2 < p; ++i) {
    const double dd[3] = {1.0, -2.0, 1.0};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) S[0](i + a, i + b) += dd[a] * dd[b];
  }
  for (int i = 0; i + 1 < p; ++i) {
    S[1](i, i) += 1.0, S[1](i + 1, i + 1) += 1.0, S[1](i, i + 1) -= 1.0, S[1](i + 1, i) -= 1.0;
  }
  for (int i = 0; i < p; ++i)
    for (int k = 0; k < p; ++k) S[2](i, k) = (i == k ? 1.0 : 0.0) - 1.0 / p;
  return S;
}

}  // namespace

TEST_CASE("params_from_matrix: theta back from matrix(theta)") {
  check_exact("diagonal_pdmat(4)", lp::diagonal_pdmat(4), {0.3, -1.2, 2.0, 0.0}, 1e-15);
  check_exact("ar1_pdmat(5)", lp::ar1_pdmat(5), {0.7}, 1e-14);
  check_exact("ar1_pdmat(5), negative", lp::ar1_pdmat(5), {-1.9}, 1e-13);
  check_exact("compound_symmetry_pdmat(4)", lp::compound_symmetry_pdmat(4), {-0.3}, 1e-14);
  check_exact("compound_symmetry_pdmat(2)", lp::compound_symmetry_pdmat(2), {2.5}, 1e-13);
  check_exact("toeplitz_pdmat(5)", lp::toeplitz_pdmat(5), {0.6, -0.4, 0.3, -0.2}, 1e-13);
  check_exact("unstructured_pdmat<log_cholesky>(4)", lp::unstructured_pdmat<bj::log_cholesky_bijector>(4),
              {0.2, 0.5, -0.3, -0.4, 0.1, 0.6, 0.3, -0.2, 0.4, -0.5}, 1e-13);
  check_exact("unstructured_pdmat<matrix_log>(3)", lp::unstructured_pdmat<bj::matrix_log_bijector>(3),
              {0.2, 0.5, -0.3, -0.4, 0.1, 0.6}, 1e-13);
  check_exact("unstructured_corr_pdmat(4)", lp::unstructured_corr_pdmat(4), {0.5, -0.3, 0.2, 0.8, -0.6, 0.1}, 1e-13);
  using scaled_ar1 = lp::unit_diagonal_scaled_pdmat<lp::ar1_pdmat>;
  using scaled_corr = lp::unit_diagonal_scaled_pdmat<lp::unstructured_corr_pdmat>;
  check_exact("unit_diagonal_scaled<ar1>(4)", scaled_ar1(lp::ar1_pdmat(4)), {0.2, -0.5, 1.1, 0.3, 0.6}, 1e-14);
  check_exact("unit_diagonal_scaled<ar1>(4), common", scaled_ar1(lp::ar1_pdmat(4), true), {0.7, -0.6}, 1e-14);
  check_exact("unit_diagonal_scaled<corr>(3)", scaled_corr(lp::unstructured_corr_pdmat(3)), {0.2, -0.5, 1.1, 0.4, -0.3, 0.6},
              1e-13);
  check_exact("reduced_rank_psdmat(5, 2)", lp::reduced_rank_psdmat(5, 2), {0.3, 0.5, -0.2, 0.4, -0.6, 0.2, 0.1, -0.3, 0.7},
              1e-12);
  check_exact("weighted_sum_psdmat(3 x 6 x 6)", lp::weighted_sum_psdmat(penalty_matrices()), {1.5, -0.5, 0.3}, 1e-12);
}

TEST_CASE("params_from_matrix: distance of the fit when the structure is not inverted") {
  // factor analysis: a matrix of the structure, which principal axis
  // factoring approaches without reaching it
  const lp::factor_analytic_pdmat fa(8, 2);
  std::vector<double> th(fa.n_params());
  for (int k = 0; k < fa.n_params(); ++k) th[k] = 0.4 * std::sin(1.7 * k + 0.3);
  const double fa_own = matrix_distance(fa, matrix_of(fa, th));
  CHECK(fa_own <= 0.05);
  // the same on a matrix outside the structure
  const double fa_other = matrix_distance(fa, test_matrix(8, 7));
  if (print_errors())
    std::printf(
        "params_from_matrix distance max|M(theta_hat) - M|/max|M|: factor_analytic(8, 2) own %.2e, other %.2e\n",
        fa_own, fa_other);
}

TEST_CASE("params_from_matrix: degenerate targets stay inside the domain") {
  // a rank-one 4 x 4 matrix with a zero diagonal entry
  const int p = 4;
  std::vector<double> m(lp::packed_size(p));
  const double v[p] = {1.0, 2.0, 0.0, -1.0};
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) m[lp::packed_index(i, j)] = v[i] * v[j];
  auto finite = [](const std::vector<double>& t) {
    for (double x : t)
      if (!std::isfinite(x)) return false;
    return true;
  };
  const double rho_max = lp::detail_pdmat::start_rho_max;
  const auto t_ar1 = params_of(lp::ar1_pdmat(p), m);
  const auto t_cs = params_of(lp::compound_symmetry_pdmat(p), m);
  const auto t_toep = params_of(lp::toeplitz_pdmat(p), m);
  CHECK(finite(t_ar1));
  CHECK(finite(t_cs));
  CHECK(finite(t_toep));
  CHECK(std::fabs(std::tanh(t_ar1[0])) <= rho_max + 1e-15);
  for (double x : t_toep) CHECK(std::fabs(std::tanh(x)) <= rho_max + 1e-15);
  CHECK(finite(params_of(lp::diagonal_pdmat(p), m)));
  // the eigenvalue floor: the smallest eigenvalue of M(theta_hat) relative to the largest
  auto spread = [&](const auto& s) {
    std::vector<double> lam, V;
    const auto back = matrix_of(s, params_of(s, m));
    lp::detail_pdmat::eigen_decreasing(p, back.data(), lam, V);
    return lam[p - 1] / lam[0];
  };
  const double floor = lp::detail_pdmat::start_eigen_floor;
  CHECK(spread(lp::unstructured_pdmat<bj::log_cholesky_bijector>(p)) >= 0.5 * floor);
  CHECK(spread(lp::unstructured_pdmat<bj::matrix_log_bijector>(p)) >= 0.5 * floor);
  CHECK(spread(lp::unstructured_corr_pdmat(p)) >= 0.1 * floor);
  CHECK(spread(lp::unit_diagonal_scaled_pdmat<lp::unstructured_corr_pdmat>(lp::unstructured_corr_pdmat(p))) >= 1e-3 * floor);
  CHECK(finite(params_of(lp::factor_analytic_pdmat(p, 2), m)));
  CHECK(finite(params_of(lp::reduced_rank_psdmat(p, 2), m)));
  // unit correlations
  std::vector<double> ones(lp::packed_size(p), 1.0);
  CHECK(std::fabs(std::tanh(params_of(lp::ar1_pdmat(p), ones)[0]) - rho_max) <= 1e-15);
  CHECK(finite(params_of(lp::compound_symmetry_pdmat(p), ones)));
  CHECK(finite(params_of(lp::toeplitz_pdmat(p), ones)));
}
