// The Eigen backend of laplace::dense: the only translation unit of Laplace
// that includes Eigen. It is compiled as C++ in every build (never by nvcc),
// with EIGEN_MPL2_ONLY and EIGEN_DONT_PARALLELIZE set by CMake. The matrices
// of the caller are mapped with Eigen::Map, without copies; the solvers copy
// their input into their own storage, so an output may alias an input.

#include "laplace/dense/detail/dense_backend.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace laplace::dense::detail {
namespace {

using matrix = Eigen::MatrixXd;
using const_matrix_map = Eigen::Map<const matrix, 0, Eigen::OuterStride<>>;
using matrix_map = Eigen::Map<matrix, 0, Eigen::OuterStride<>>;
using vector_map = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<>>;

const_matrix_map map(const_matrix_ref a) {
  return const_matrix_map(a.data, a.rows, a.cols, Eigen::OuterStride<>(a.ld));
}
matrix_map map(matrix_ref a) { return matrix_map(a.data, a.rows, a.cols, Eigen::OuterStride<>(a.ld)); }
vector_map map(vector_ref v) { return vector_map(v.data, v.size, Eigen::InnerStride<>(v.inc)); }

std::string shape(int rows, int cols) { return std::to_string(rows) + " x " + std::to_string(cols); }

[[noreturn]] void wrong_dimensions(const char* function, const std::string& what) {
  throw std::invalid_argument(std::string("laplace::dense::") + function + ": " + what);
}

template <class Ref>
void require_shape(const char* function, const char* name, const Ref& a, int rows, int cols) {
  if (a.rows != rows || a.cols != cols)
    wrong_dimensions(function, std::string(name) + " is " + shape(a.rows, a.cols) + ", expected " + shape(rows, cols));
}

void require_square(const char* function, const char* name, const const_matrix_ref& a) {
  if (a.rows != a.cols) wrong_dimensions(function, std::string(name) + " is " + shape(a.rows, a.cols) + ", not square");
}

void require_size(const char* function, const char* name, int size, int expected) {
  if (size != expected)
    wrong_dimensions(function, std::string(name) + " has " + std::to_string(size) + " entries, expected " +
                                   std::to_string(expected));
}

// The lower triangle of a square matrix, the part a symmetric solver reads.
bool lower_is_finite(const const_matrix_map& a) {
  for (Eigen::Index j = 0; j < a.cols(); ++j)
    for (Eigen::Index i = j; i < a.rows(); ++i)
      if (!std::isfinite(a(i, j))) return false;
  return true;
}

bool upper_is_finite(const const_matrix_map& a) {
  for (Eigen::Index j = 0; j < a.cols(); ++j)
    for (Eigen::Index i = 0; i <= j; ++i)
      if (!std::isfinite(a(i, j))) return false;
  return true;
}

dense_decomposition_status failure(dense_failure reason) { return {false, 0, reason}; }
dense_decomposition_status success(int rank) { return {true, rank, dense_failure::none}; }

// The relative rank tolerance: max(m, n) eps unless the caller gave one.
double rank_tolerance(double relative_tolerance, int m, int n) {
  return relative_tolerance >= 0.0 ? relative_tolerance
                                   : std::max(m, n) * std::numeric_limits<double>::epsilon();
}

// The number of s_i above tol s_1 (s in decreasing order, s_1 >= 0).
int rank_from_decreasing(const Eigen::VectorXd& s, double tol) {
  if (s.size() == 0 || !(s(0) > 0.0)) return 0;
  const double threshold = tol * s(0);
  int r = 0;
  while (r < s.size() && s(r) > threshold) ++r;
  return r;
}

}  // namespace

dense_decomposition_status cholesky(const_matrix_ref A, matrix_ref L) {
  require_square("cholesky_decomposition", "A", A);
  require_shape("cholesky_decomposition", "L", L, A.rows, A.rows);
  const const_matrix_map a = map(A);
  if (!lower_is_finite(a)) return failure(dense_failure::non_finite_input);
  const Eigen::LLT<matrix> llt(a);  // reads the lower triangle
  if (llt.info() != Eigen::Success) return failure(dense_failure::not_positive_definite);
  map(L) = llt.matrixL();  // the strictly upper part is set to zero
  return success(A.rows);
}

dense_decomposition_status triangular_solve(const_matrix_ref T, matrix_ref B, dense_triangle triangle,
                                            dense_operation operation) {
  require_square("triangular_solve", "T", T);
  if (B.rows != T.rows)
    wrong_dimensions("triangular_solve", "B has " + std::to_string(B.rows) + " rows, T is " + shape(T.rows, T.cols));
  const const_matrix_map t = map(T);
  matrix_map b = map(B);
  const bool lower = triangle == dense_triangle::lower;
  if (!(lower ? lower_is_finite(t) : upper_is_finite(t)) || !b.allFinite())
    return failure(dense_failure::non_finite_input);
  for (Eigen::Index i = 0; i < t.rows(); ++i)
    if (t(i, i) == 0.0) return failure(dense_failure::singular);
  const bool transposed = operation == dense_operation::transpose;
  if (lower && !transposed) t.triangularView<Eigen::Lower>().solveInPlace(b);
  if (lower && transposed) t.transpose().triangularView<Eigen::Upper>().solveInPlace(b);
  if (!lower && !transposed) t.triangularView<Eigen::Upper>().solveInPlace(b);
  if (!lower && transposed) t.transpose().triangularView<Eigen::Lower>().solveInPlace(b);
  return success(T.rows);
}

// (L L^T)^-1 B in place: L y = b, then L^T x = y.
dense_decomposition_status cholesky_solve(const_matrix_ref L, matrix_ref B) {
  require_square("cholesky_solve", "L", L);
  if (B.rows != L.rows)
    wrong_dimensions("cholesky_solve", "B has " + std::to_string(B.rows) + " rows, L is " + shape(L.rows, L.cols));
  const dense_decomposition_status first = triangular_solve(L, B, dense_triangle::lower, dense_operation::identity);
  if (!first.success) return first;
  return triangular_solve(L, B, dense_triangle::lower, dense_operation::transpose);
}

// kappa_1(A) = |A|_1 |A^-1|_1, with |A^-1|_1 estimated by Higham's
// refinement of Hager's method (Higham 1988, ACM TOMS 14, Algorithm 4.1; the
// iteration of LAPACK's dlacn2): at most five solves of the power-like
// iteration on sign vectors plus one with the alternating vector
// x_i = (-1)^i (1 + i / (n - 1)). A^-1 is symmetric, so the transposed solves
// of the general method are the same solves. Each solve is two triangular
// solves with L, O(n^2).
double cholesky_condition_estimate(const_matrix_ref A, const_matrix_ref L) {
  require_square("cholesky_condition_estimate", "A", A);
  require_shape("cholesky_condition_estimate", "L", L, A.rows, A.rows);
  const int n = A.rows;
  if (n == 0) return 0.0;
  const const_matrix_map a = map(A), l = map(L);
  if (!lower_is_finite(a)) return std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < n; ++i)
    if (!(std::isfinite(l(i, i)) && l(i, i) != 0.0)) return std::numeric_limits<double>::infinity();
  // |A|_1 of the symmetric A from its lower triangle
  double a_norm = 0.0;
  for (int j = 0; j < n; ++j) {
    double column = 0.0;
    for (int i = 0; i < n; ++i) column += std::abs(i >= j ? a(i, j) : a(j, i));
    a_norm = std::max(a_norm, column);
  }
  const auto lower = l.triangularView<Eigen::Lower>();
  auto solve = [&](Eigen::VectorXd& x) {
    lower.solveInPlace(x);
    lower.transpose().solveInPlace(x);
  };
  auto sign = [](double v) { return v >= 0.0 ? 1.0 : -1.0; };
  Eigen::VectorXd x = Eigen::VectorXd::Constant(n, 1.0 / n);
  solve(x);
  double estimate = x.lpNorm<1>();
  if (n > 1) {
    Eigen::VectorXd xi = x.unaryExpr(sign);
    x = xi;
    solve(x);
    Eigen::Index j = 0;
    x.cwiseAbs().maxCoeff(&j);
    for (int iteration = 2; iteration <= 5; ++iteration) {
      x.setZero();
      x(j) = 1.0;
      solve(x);
      const double previous = estimate;
      estimate = x.lpNorm<1>();
      const Eigen::VectorXd xi_new = x.unaryExpr(sign);
      if (xi_new == xi || estimate <= previous) {
        estimate = std::max(estimate, previous);
        break;
      }
      xi = xi_new;
      x = xi;
      solve(x);
      const Eigen::Index j_last = j;
      x.cwiseAbs().maxCoeff(&j);
      if (x(j_last) == std::abs(x(j))) break;
    }
    for (int i = 0; i < n; ++i) x(i) = (i % 2 == 0 ? 1.0 : -1.0) * (1.0 + double(i) / (n - 1));
    solve(x);
    estimate = std::max(estimate, 2.0 * x.lpNorm<1>() / (3.0 * n));
  }
  return a_norm * estimate;
}

dense_decomposition_status pivoted_qr(const_matrix_ref A, matrix_ref Q, matrix_ref R, int_vector_ref perm,
                                      double relative_tolerance) {
  const int m = A.rows, n = A.cols, p = std::min(m, n);
  if (Q.rows != m || (Q.cols != p && Q.cols != m))
    wrong_dimensions("pivoted_qr_decomposition",
                     "Q is " + shape(Q.rows, Q.cols) + ", expected " + shape(m, p) + " or " + shape(m, m));
  require_shape("pivoted_qr_decomposition", "R", R, Q.cols, n);
  require_size("pivoted_qr_decomposition", "perm", perm.size, n);
  const const_matrix_map a = map(A);
  if (!a.allFinite()) return failure(dense_failure::non_finite_input);
  const Eigen::ColPivHouseholderQR<matrix> qr(a);
  matrix q = matrix::Identity(m, Q.cols);
  q.applyOnTheLeft(qr.householderQ());
  map(Q) = q;
  // R: the upper triangle of the first Q.cols rows (rows beyond min(m, n) are zero)
  matrix r = matrix::Zero(Q.cols, n);
  r.topRows(p) = qr.matrixQR().topRows(p).triangularView<Eigen::Upper>();
  map(R) = r;
  for (int j = 0; j < n; ++j) perm.data[j * perm.inc] = qr.colsPermutation().indices()(j);
  Eigen::VectorXd diagonal(p);
  for (int k = 0; k < p; ++k) diagonal(k) = std::abs(qr.matrixQR()(k, k));
  return success(rank_from_decreasing(diagonal, rank_tolerance(relative_tolerance, m, n)));
}

dense_decomposition_status svd(const_matrix_ref A, matrix_ref U, vector_ref s, matrix_ref V,
                               double relative_tolerance) {
  const int m = A.rows, n = A.cols, p = std::min(m, n);
  if (U.rows != m || (U.cols != p && U.cols != m))
    wrong_dimensions("svd_decomposition",
                     "U is " + shape(U.rows, U.cols) + ", expected " + shape(m, p) + " or " + shape(m, m));
  if (V.rows != n || (V.cols != p && V.cols != n))
    wrong_dimensions("svd_decomposition",
                     "V is " + shape(V.rows, V.cols) + ", expected " + shape(n, p) + " or " + shape(n, n));
  require_size("svd_decomposition", "s", s.size, p);
  const const_matrix_map a = map(A);
  if (!a.allFinite()) return failure(dense_failure::non_finite_input);
  const unsigned options = (U.cols == m ? Eigen::ComputeFullU : Eigen::ComputeThinU) |
                           (V.cols == n ? Eigen::ComputeFullV : Eigen::ComputeThinV);
  const Eigen::BDCSVD<matrix> decomposition(a, options);
  if (decomposition.info() != Eigen::Success) return failure(dense_failure::no_convergence);
  map(U) = decomposition.matrixU();
  map(V) = decomposition.matrixV();
  const Eigen::VectorXd values = decomposition.singularValues();
  map(s) = values;
  return success(rank_from_decreasing(values, rank_tolerance(relative_tolerance, m, n)));
}

dense_decomposition_status symmetric_eigen(const_matrix_ref A, vector_ref w, matrix_ref V) {
  require_square("symmetric_eigen_decomposition", "A", A);
  require_size("symmetric_eigen_decomposition", "w", w.size, A.rows);
  require_shape("symmetric_eigen_decomposition", "V", V, A.rows, A.rows);
  const const_matrix_map a = map(A);
  if (!lower_is_finite(a)) return failure(dense_failure::non_finite_input);
  const Eigen::SelfAdjointEigenSolver<matrix> solver(a, Eigen::ComputeEigenvectors);  // reads the lower triangle
  if (solver.info() != Eigen::Success) return failure(dense_failure::no_convergence);
  map(w) = solver.eigenvalues();
  map(V) = solver.eigenvectors();
  return success(A.rows);
}

// With B = L L^T: C = L^-1 A L^-T is symmetric with the same eigenvalues, and
// C Y = Y diag(w) gives V = L^-T Y, so that V^T B V = Y^T Y = I.
dense_decomposition_status generalized_symmetric_eigen(const_matrix_ref A, const_matrix_ref B, vector_ref w,
                                                       matrix_ref V) {
  require_square("generalized_symmetric_eigen_decomposition", "A", A);
  require_shape("generalized_symmetric_eigen_decomposition", "B", B, A.rows, A.rows);
  require_size("generalized_symmetric_eigen_decomposition", "w", w.size, A.rows);
  require_shape("generalized_symmetric_eigen_decomposition", "V", V, A.rows, A.rows);
  const const_matrix_map a = map(A), b = map(B);
  if (!lower_is_finite(a) || !lower_is_finite(b)) return failure(dense_failure::non_finite_input);
  const Eigen::LLT<matrix> llt(b);
  if (llt.info() != Eigen::Success) return failure(dense_failure::not_positive_definite);
  matrix c = a.selfadjointView<Eigen::Lower>();
  llt.matrixL().solveInPlace(c);  // L^-1 A
  c.transposeInPlace();           // A L^-T
  llt.matrixL().solveInPlace(c);  // L^-1 A L^-T
  const Eigen::SelfAdjointEigenSolver<matrix> solver(c, Eigen::ComputeEigenvectors);
  if (solver.info() != Eigen::Success) return failure(dense_failure::no_convergence);
  matrix v = solver.eigenvectors();
  llt.matrixU().solveInPlace(v);  // L^-T Y
  map(w) = solver.eigenvalues();
  map(V) = v;
  return success(A.rows);
}

dense_decomposition_status null_space(const_matrix_ref C, double* V, double relative_tolerance) {
  const int m = C.rows, n = C.cols;
  matrix_map v(V, n, n, Eigen::OuterStride<>(n > 0 ? n : 1));
  if (m == 0) {
    v.setIdentity();
    return success(0);
  }
  const const_matrix_map c = map(C);
  if (!c.allFinite()) return failure(dense_failure::non_finite_input);
  // JacobiSVD: C has few rows, and its QR preconditioner gives the full V.
  const Eigen::JacobiSVD<matrix> decomposition(c, Eigen::ComputeFullV);
  if (decomposition.info() != Eigen::Success) return failure(dense_failure::no_convergence);
  v = decomposition.matrixV();
  const Eigen::VectorXd values = decomposition.singularValues();
  return success(rank_from_decreasing(values, rank_tolerance(relative_tolerance, m, n)));
}

}  // namespace laplace::dense::detail
