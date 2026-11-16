// distrib_penalty<M> (include/laplace/penalty/) over mvgaussian2_distrib:
//  - the descriptors and the normalization class (proper over a pdmat
//    structure, intrinsic over a constant_null_space_psdmat);
//  - the twin pen = -logpdf, with and without a prior location mu_0;
//  - every derivative block (J, K) in (beta, theta), J <= 2, J + K <= 3, against
//    central differences of the block below it (normwise error);
//  - the normalization: the integral of exp(-pen) over beta is 1 (proper, 2-d)
//    and over the range space of Q is 1 (intrinsic, rank 2 in 3-d), by the
//    trapezoid rule, which converges geometrically on a Gaussian;
//  - the null space is flat: pen(beta + c 1) = pen(beta).
// The proper case runs inside a Kokkos kernel (a named function with external
// linkage) and on the host; the intrinsic case runs on the host, since
// weighted_sum_psdmat is host only.
#include <doctest/doctest.h>

#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/pdmat/unstructured.hpp>
#include <laplace/pdmat/weighted_sum.hpp>
#include <laplace/penalty/distrib_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace penalty_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
namespace lpen = laplace::penalty;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using ws_t = lp::weighted_sum_psdmat;
using host_matrix = ws_t::host_matrix;
using proper_pen = lpen::distrib_penalty<ld::mvgaussian2_distrib<log_chol>>;
using intrinsic_pen = lpen::distrib_penalty<ld::mvgaussian2_distrib<ws_t>>;

static_assert(lpen::penalty<proper_pen> && lpen::penalty<intrinsic_pen>);
static_assert(proper_pen::normalization == lpen::penalty_normalization::proper);
static_assert(intrinsic_pen::normalization == lpen::penalty_normalization::intrinsic);
static_assert(!proper_pen::q_linear_in_theta && intrinsic_pen::q_linear_in_theta);
static_assert(proper_pen::kinks == lpen::penalty_kink_geometry::none &&
              proper_pen::prox == lpen::penalty_prox_kind::none &&
              proper_pen::structure == lpen::penalty_structure::coupled &&
              proper_pen::scale_mixture == lpen::penalty_scale_mixture::none &&
              proper_pen::beta_regularity == ld::infinitely_smooth &&
              proper_pen::theta_regularity == ld::infinitely_smooth);

constexpr int max_out = 15;  // packed_size(5)
constexpr int n_checks = 9;

KOKKOS_INLINE_FUNCTION int out_size(int J, int d) { return J == 0 ? 1 : (J == 1 ? d : d * (d + 1) / 2); }

// the block (J, K) of the penalty at (theta, beta) into out
// the checks also run on the host-only intrinsic penalty: no nvcc host/device call warning
LAPLACE_PENALTY_NO_EXEC_CHECK
template <class P, int J, int K>
KOKKOS_INLINE_FUNCTION bool block(const P& pen, const double* theta, const double* beta, const int* idx, double* out,
                                  double* w) {
  const auto st = pen.theta_state(theta, w);
  double* ws = w + pen.state_size();
  if constexpr (J == 0 && K == 0) {
    out[0] = pen.value(st, beta, ws);
    return true;
  } else {
    return pen.template value_dbeta_dk<J, K>(st, beta, idx, out, ws);
  }
}

// normwise error of the block (J, K) against the central difference of the block
// below it: in theta_idx[K-1] for K >= 1, in beta_c for K = 0
LAPLACE_PENALTY_NO_EXEC_CHECK
template <class P, int J, int K>
KOKKOS_INLINE_FUNCTION double fd_error(const P& pen, const double* theta, const double* beta, int q, double* w) {
  const int d = pen.dim();
  const int tab[6] = {0, 1, 1, 3, 3, 4};
  int idx[6];
  for (int k = 0; k < 6; ++k) idx[k] = tab[k] < q ? tab[k] : q - 1;
  double a[max_out], p1[max_out], m1[max_out];
  if (!block<P, J, K>(pen, theta, beta, idx, a, w)) return 0.0;
  const int n = out_size(J, d);
  double scale = 1.0, err = 0.0;
  for (int i = 0; i < n; ++i) scale = Kokkos::fmax(scale, Kokkos::fabs(a[i]));
  double th[16], bt[8];
  for (int k = 0; k < q; ++k) th[k] = theta[k];
  for (int k = 0; k < d; ++k) bt[k] = beta[k];
  if constexpr (K >= 1) {
    const int j = idx[K - 1];
    const double h = 1e-5 * Kokkos::fmax(1.0, Kokkos::fabs(theta[j]));
    th[j] = theta[j] + h;
    block<P, J, K - 1>(pen, th, beta, idx, p1, w);
    th[j] = theta[j] - h;
    block<P, J, K - 1>(pen, th, beta, idx, m1, w);
    for (int i = 0; i < n; ++i) err = Kokkos::fmax(err, Kokkos::fabs(a[i] - (p1[i] - m1[i]) / (2.0 * h)));
  } else if constexpr (J >= 1) {
    for (int c = 0; c < d; ++c) {
      const double h = 1e-5 * Kokkos::fmax(1.0, Kokkos::fabs(beta[c]));
      bt[c] = beta[c] + h;
      block<P, J - 1, 0>(pen, theta, bt, idx, p1, w);
      bt[c] = beta[c] - h;
      block<P, J - 1, 0>(pen, theta, bt, idx, m1, w);
      bt[c] = beta[c];
      if constexpr (J == 1) {
        err = Kokkos::fmax(err, Kokkos::fabs(a[c] - (p1[0] - m1[0]) / (2.0 * h)));
      } else {
        for (int r = 0; r < d; ++r) {
          const int e = r >= c ? r * (r + 1) / 2 + c : c * (c + 1) / 2 + r;
          err = Kokkos::fmax(err, Kokkos::fabs(a[e] - (p1[r] - m1[r]) / (2.0 * h)));
        }
      }
    }
  }
  return err / scale;
}

// res[0]: |value + logpdf| (zero when exact); res[1..8]: the finite-difference
// errors of the blocks (0,1) (0,2) (0,3) (1,0) (2,0) (1,1) (1,2) (2,1)
LAPLACE_PENALTY_NO_EXEC_CHECK
template <class P>
KOKKOS_INLINE_FUNCTION void all_checks(const P& pen, const double* theta, const double* beta, int q, double* w,
                                       double* res) {
  const int d = pen.dim();
  double zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const double* loc = pen.mu0 != nullptr ? pen.mu0 : zero;
  const auto st = pen.theta_state(theta, w);
  double* ws = w + pen.state_size();
  res[0] = Kokkos::fabs(pen.value(st, beta, ws) + pen.distribution.logpdf(st, beta, loc));
  (void)d;
  res[1] = fd_error<P, 0, 1>(pen, theta, beta, q, w);
  res[2] = fd_error<P, 0, 2>(pen, theta, beta, q, w);
  res[3] = fd_error<P, 0, 3>(pen, theta, beta, q, w);
  res[4] = fd_error<P, 1, 0>(pen, theta, beta, q, w);
  res[5] = fd_error<P, 2, 0>(pen, theta, beta, q, w);
  res[6] = fd_error<P, 1, 1>(pen, theta, beta, q, w);
  res[7] = fd_error<P, 1, 2>(pen, theta, beta, q, w);
  res[8] = fd_error<P, 2, 1>(pen, theta, beta, q, w);
}

// the proper case in one kernel thread, without and with a prior location
void run_proper_kernel(Kokkos::View<double*> res, Kokkos::View<double*> work) {
  Kokkos::parallel_for(
      "penalty_proper", 1, KOKKOS_LAMBDA(int) {
        const log_chol s(3);
        const ld::mvgaussian2_distrib<log_chol> m(s);
        const double theta[6] = {0.3, -0.2, 0.4, 0.1, -0.3, 0.2};
        const double beta[3] = {0.7, -1.1, 0.4};
        const double mu0[3] = {0.2, 0.5, -0.6};
        double* w = &work(0);
        const proper_pen a(m), b(m, mu0);
        all_checks(a, theta, beta, 6, w, &res(0));
        all_checks(b, theta, beta, 6, w, &res(n_checks));
        res(2 * n_checks) = a.weak_convexity(theta);
      });
}

host_matrix gram_of_differences(int d, int k) {
  std::vector<std::vector<double>> r(d, std::vector<double>(d, 0.0));
  for (int i = 0; i < d; ++i) r[i][i] = 1.0;
  for (int s = 0; s < k; ++s) {
    std::vector<std::vector<double>> n(r.size() - 1, std::vector<double>(d));
    for (std::size_t i = 0; i + 1 < r.size(); ++i)
      for (int j = 0; j < d; ++j) n[i][j] = r[i + 1][j] - r[i][j];
    r = n;
  }
  host_matrix m("S", d, d);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
      for (std::size_t t = 0; t < r.size(); ++t) m(i, j) += r[t][i] * r[t][j];
  return m;
}

void check_blocks(const double* res, double tol_fd) {
  CHECK(res[0] == 0.0);
  for (int k = 1; k < n_checks; ++k) {
    CAPTURE(k);
    CHECK(res[k] < tol_fd);
  }
}

}  // namespace penalty_test

TEST_CASE("distrib_penalty over mvgaussian2: twin and finite differences, proper, in a kernel and on the host") {
  using namespace penalty_test;
  const log_chol s(3);
  const ld::mvgaussian2_distrib<log_chol> m(s);
  const double theta[6] = {0.3, -0.2, 0.4, 0.1, -0.3, 0.2};
  const double beta[3] = {0.7, -1.1, 0.4};
  const double mu0[3] = {0.2, 0.5, -0.6};
  const proper_pen a(m), b(m, mu0);
  std::vector<double> w(a.state_size() + a.scratch_size());
  Kokkos::View<double*> res("res", 2 * n_checks + 1), work("work", w.size());
  run_proper_kernel(res, work);
  auto h = Kokkos::create_mirror_view(res);
  Kokkos::deep_copy(h, res);
  check_blocks(&h(0), 1e-6);
  check_blocks(&h(n_checks), 1e-6);
  CHECK(h(2 * n_checks) == 0.0);

  // the same on the host, and the kernel agrees with it
  double r0[n_checks], r1[n_checks];
  all_checks(a, theta, beta, 6, w.data(), r0);
  all_checks(b, theta, beta, 6, w.data(), r1);
  check_blocks(r0, 1e-6);
  check_blocks(r1, 1e-6);
  for (int k = 1; k < n_checks; ++k) {
    CHECK(std::fabs(r0[k] - h(k)) < 1e-9);
    CHECK(std::fabs(r1[k] - h(n_checks + k)) < 1e-9);
  }
}

TEST_CASE("distrib_penalty over mvgaussian2: twin and finite differences, intrinsic, on the host") {
  using namespace penalty_test;
  const ws_t s(std::vector<host_matrix>{gram_of_differences(5, 1), gram_of_differences(5, 2)});
  const ld::mvgaussian2_distrib<ws_t> m(s);
  const double theta[2] = {0.3, -0.4};
  const double beta[5] = {0.7, -1.1, 0.4, 0.9, -0.2};
  const double mu0[5] = {0.2, 0.5, -0.6, 0.0, 0.3};
  const intrinsic_pen a(m), b(m, mu0);
  CHECK(a.weak_convexity(theta) == 0.0);
  std::vector<double> w(a.state_size() + a.scratch_size());
  double r0[n_checks], r1[n_checks];
  all_checks(a, theta, beta, 2, w.data(), r0);
  all_checks(b, theta, beta, 2, w.data(), r1);
  check_blocks(r0, 1e-6);
  check_blocks(r1, 1e-6);
}

TEST_CASE("distrib_penalty over mvgaussian2: exp(-pen) integrates to one") {
  using namespace penalty_test;
  {  // proper, d = 2, with a prior location
    const log_chol s(2);
    const ld::mvgaussian2_distrib<log_chol> m(s);
    const double theta[3] = {0.1, 0.2, -0.1}, mu0[2] = {0.3, -0.2};
    const proper_pen p(m, mu0);
    std::vector<double> w(p.state_size() + p.scratch_size());
    const auto st = p.theta_state(theta, w.data());
    double* ws = w.data() + p.state_size();
    const double h = 0.03, L = 9.0;
    double sum = 0.0;
    for (double x = -L; x <= L; x += h)
      for (double y = -L; y <= L; y += h) {
        const double beta[2] = {mu0[0] + x, mu0[1] + y};
        sum += std::exp(-p.value(st, beta, ws));
      }
    CHECK(std::fabs(sum * h * h - 1.0) < 1e-9);
  }
  {  // intrinsic, d = 3, rank 2: integrate over the range space, flat along 1
    const ws_t s(std::vector<host_matrix>{gram_of_differences(3, 1), gram_of_differences(3, 2)});
    REQUIRE(s.rank() == 2);
    const ld::mvgaussian2_distrib<ws_t> m(s);
    const double theta[2] = {0.2, -0.3};
    const intrinsic_pen p(m);
    std::vector<double> w(p.state_size() + p.scratch_size());
    const auto st = p.theta_state(theta, w.data());
    double* ws = w.data() + p.state_size();
    const double r2 = std::sqrt(2.0), r6 = std::sqrt(6.0);
    const double h = 0.03, L = 9.0;
    double sum = 0.0;
    for (double x = -L; x <= L; x += h)
      for (double y = -L; y <= L; y += h) {
        const double beta[3] = {x / r2 + y / r6, -x / r2 + y / r6, -2.0 * y / r6};
        sum += std::exp(-p.value(st, beta, ws));
      }
    CHECK(std::fabs(sum * h * h - 1.0) < 1e-9);
    const double b0[3] = {0.4, -0.7, 1.3}, b1[3] = {0.4 + 2.5, -0.7 + 2.5, 1.3 + 2.5};
    CHECK(std::fabs(p.value(st, b0, ws) - p.value(st, b1, ws)) < 1e-12);
  }
}
