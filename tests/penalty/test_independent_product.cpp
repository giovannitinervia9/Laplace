// univariate_as_multivariate<D> and independent_product_distrib<M...>
// (include/laplace/distrib/), and their penalties (include/laplace/penalty/):
//  - univariate_as_multivariate: the log-density and every derivative block
//    (J in y, K in theta, A in mu) are the components of D's own tensors;
//  - the product (lasso, Student by group, and a heterogeneous product of enet,
//    two iid mvgaussian2 copies with theta by copy, and pseudohuber): the twin
//    log p = sum of the components' log-densities; every dense block (J, K, A)
//    with J <= 2, K <= 3, A <= 1 against central differences of the block
//    below it (normwise error); the dense route equals the block route placed
//    on the diagonal blocks, zero elsewhere; a theta tuple spanning two sets
//    is zero;
//  - distrib_penalty over the product: pen = -logpdf, the derivatives in beta
//    are minus the block route; the descriptors; rho of the Student penalty
//    against the least value of pen'' on a grid, and the product's rho as the
//    largest over the sets;
//  - the normalization: exp(-pen) of a 2-d product (pseudohuber x gaussian2)
//    integrates to one (trapezoid rule).
// The Student and heterogeneous products also run inside a Kokkos kernel (a
// named function with external linkage) and must agree with the host.
#include <doctest/doctest.h>

#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/pdmat/unstructured.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace product_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
namespace lpen = laplace::penalty;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
template <class D>
using uam = ld::univariate_as_multivariate<D>;
using lasso_t = ld::independent_product_distrib<uam<ld::laplace1_distrib>>;
using student_t = ld::independent_product_distrib<uam<ld::student_t1_distrib>>;
using mixed_t =
    ld::independent_product_distrib<uam<ld::enet_distrib>, ld::mvgaussian2_distrib<log_chol>, uam<ld::pseudohuber_distrib>>;

static_assert(ld::multivariate_distrib<uam<ld::student_t2_distrib>> && ld::multivariate_distrib<mixed_t>);
static_assert(lpen::penalty<lpen::distrib_penalty<lasso_t>> && lpen::penalty<lpen::distrib_penalty<mixed_t>>);
static_assert(mixed_t::n_params == 1 + 2 + 1 + 2 && mixed_t::params[0].kind == ld::distrib_mv_param_kind::vector &&
              mixed_t::params[3].kind == ld::distrib_mv_param_kind::matrix &&
              mixed_t::params[5].kind == ld::distrib_mv_param_kind::scalar && mixed_t::params[0].regularity == 0);
using lasso_pen = lpen::distrib_penalty<lasso_t>;
using student_pen = lpen::distrib_penalty<student_t>;
using mixed_pen = lpen::distrib_penalty<mixed_t>;
static_assert(lasso_pen::normalization == lpen::penalty_normalization::proper && lasso_pen::beta_regularity == 0 &&
              lasso_pen::kinks == lpen::penalty_kink_geometry::per_coordinate &&
              lasso_pen::structure == lpen::penalty_structure::separable &&
              lasso_pen::scale_mixture == lpen::penalty_scale_mixture::exponential_variance &&
              lasso_pen::theta_regularity == ld::infinitely_smooth && !lasso_pen::q_linear_in_theta);
static_assert(student_pen::kinks == lpen::penalty_kink_geometry::none &&
              student_pen::beta_regularity == ld::infinitely_smooth &&
              student_pen::scale_mixture == lpen::penalty_scale_mixture::inverse_gamma_variance);
static_assert(mixed_pen::structure == lpen::penalty_structure::grouped && mixed_pen::beta_regularity == 0 &&
              mixed_pen::kinks == lpen::penalty_kink_geometry::per_coordinate &&
              mixed_pen::scale_mixture == lpen::penalty_scale_mixture::none);

constexpr int max_out = 64;   // dense outputs: d = 7, J = 1 and A = 1
constexpr int max_theta = 16;
constexpr int n_checks = 14;

// the dense block (J, K, A) at (theta, y, x) into out, zeros when reported zero
template <class P, int J, int K, int A>
KOKKOS_INLINE_FUNCTION bool dense(const P& p, const double* th, const double* y, const double* x, const int* idx,
                                  double* out, double* w) {
  const auto st = p.theta_state(th, w);
  double* ws = w + p.state_size();
  const int n = ld::sym_tensor_size(p.dim(), J) * ld::sym_tensor_size(p.dim(), A);
  bool ok;
  if constexpr (A == 0) ok = p.template logpdf_dy_dk<J, K>(st, y, x, idx, out, ws);
  else ok = p.template logpdf_dy_dk<J, K, A>(st, y, x, idx, out, ws);
  if (!ok)
    for (int i = 0; i < n; ++i) out[i] = 0.0;
  return ok;
}

KOKKOS_INLINE_FUNCTION double rel_error(const double* a, const double* b, int n) {
  double e = 0.0, s = 1.0;
  for (int i = 0; i < n; ++i) {
    e = Kokkos::fmax(e, Kokkos::fabs(a[i] - b[i]));
    s = Kokkos::fmax(s, Kokkos::fabs(a[i]));
  }
  return e / s;
}

// the block (J, K, A), K >= 1, for every sorted tuple over theta, against central
// differences in theta[t[K-1]] of the block (J, K - 1, A) at t[0..K-2]
template <class P, int J, int K, int A>
KOKKOS_INLINE_FUNCTION double fd_theta(const P& p, const double* th, const double* y, const double* x, int q,
                                       double* w) {
  const int n = ld::sym_tensor_size(p.dim(), J) * ld::sym_tensor_size(p.dim(), A);
  double a[max_out], p1[max_out], m1[max_out], t2[max_theta];
  int t[K];
  for (int i = 0; i < K; ++i) t[i] = 0;
  double err = 0.0;
  for (int c = 0, nc = ld::sym_tensor_size(q, K); c < nc; ++c) {
    dense<P, J, K, A>(p, th, y, x, t, a, w);
    const int j = t[K - 1];
    const double h = 1e-5 * Kokkos::fmax(1.0, Kokkos::fabs(th[j]));
    for (int i = 0; i < q; ++i) t2[i] = th[i];
    t2[j] = th[j] + h;
    dense<P, J, K - 1, A>(p, t2, y, x, t, p1, w);
    t2[j] = th[j] - h;
    dense<P, J, K - 1, A>(p, t2, y, x, t, m1, w);
    for (int i = 0; i < n; ++i) p1[i] = (p1[i] - m1[i]) / (2.0 * h);
    err = Kokkos::fmax(err, rel_error(a, p1, n));
    ld::detail_product::next_sorted_tuple(t, K, q);
  }
  return err;
}

// the block (J, 0, A) with J + A >= 1 against central differences of the block one
// order lower in mu (A = 1) or in y (A = 0), coordinate by coordinate
template <class P, int J, int A>
KOKKOS_INLINE_FUNCTION double fd_point(const P& p, const double* th, const double* y, const double* x, double* w) {
  const int d = p.dim();
  constexpr int JL = A == 1 ? J : J - 1;
  const int nl = ld::sym_tensor_size(d, JL);
  double a[max_out], p1[max_out], m1[max_out], v[8], f[max_out];
  dense<P, J, 0, A>(p, th, y, x, nullptr, a, w);
  double err = 0.0;
  for (int c = 0; c < d; ++c) {
    const double* base = A == 1 ? x : y;
    for (int i = 0; i < d; ++i) v[i] = base[i];
    const double h = 1e-5 * Kokkos::fmax(1.0, Kokkos::fabs(base[c]));
    v[c] = base[c] + h;
    dense<P, JL, 0, 0>(p, th, A == 1 ? y : v, A == 1 ? v : x, nullptr, p1, w);
    v[c] = base[c] - h;
    dense<P, JL, 0, 0>(p, th, A == 1 ? y : v, A == 1 ? v : x, nullptr, m1, w);
    // the entries of a that are derivatives in coordinate c of the lower block
    for (int i = 0; i < nl; ++i) {
      int k;
      if (A == 1) k = i * d + c;                 // (J, 0, 1): y tuple i, mu coordinate c
      else if (J == 1) k = c;                    // gradient
      else k = i >= c ? i * (i + 1) / 2 + c : c * (c + 1) / 2 + i;  // packed (max, min)
      f[i] = a[k];
      p1[i] = (p1[i] - m1[i]) / (2.0 * h);
    }
    err = Kokkos::fmax(err, rel_error(f, p1, nl));
  }
  return err;
}

// res[0]: twin error of logpdf against the sum of the components; res[1..]: finite differences
template <class P>
KOKKOS_INLINE_FUNCTION void all_checks(const P& p, const double* th, const double* y, const double* x, double twin,
                                       double* w, double* res) {
  const int q = p.theta_size();
  double v;
  dense<P, 0, 0, 0>(p, th, y, x, nullptr, &v, w);
  res[0] = Kokkos::fabs(v - twin) / Kokkos::fmax(1.0, Kokkos::fabs(twin));
  res[1] = fd_point<P, 1, 0>(p, th, y, x, w);
  res[2] = fd_point<P, 2, 0>(p, th, y, x, w);
  res[3] = fd_point<P, 0, 1>(p, th, y, x, w);
  res[4] = fd_point<P, 1, 1>(p, th, y, x, w);
  res[5] = fd_theta<P, 0, 1, 0>(p, th, y, x, q, w);
  res[6] = fd_theta<P, 0, 2, 0>(p, th, y, x, q, w);
  res[7] = fd_theta<P, 0, 3, 0>(p, th, y, x, q, w);
  res[8] = fd_theta<P, 1, 1, 0>(p, th, y, x, q, w);
  res[9] = fd_theta<P, 1, 2, 0>(p, th, y, x, q, w);
  res[10] = fd_theta<P, 2, 1, 0>(p, th, y, x, q, w);
  res[11] = fd_theta<P, 0, 1, 1>(p, th, y, x, q, w);
  res[12] = fd_theta<P, 1, 1, 1>(p, th, y, x, q, w);
  res[13] = fd_theta<P, 0, 2, 1>(p, th, y, x, q, w);
}

// the Student product: 5 copies in 2 sets, theta = (sigma_0, sigma_1, nu_0, nu_1)
struct student_case {
  int set_of[5] = {0, 1, 0, 1, 1};
  double theta[4] = {0.8, 1.7, 3.0, 9.0};
  double y[5] = {0.7, -1.1, 0.4, 2.3, -0.2};
  double x[5] = {0.1, 0.3, -0.5, 0.2, 0.0};
  KOKKOS_INLINE_FUNCTION student_t make() const {
    return student_t(ld::independent_product_component<uam<ld::student_t1_distrib>>{{}, 5, 2, set_of});
  }
  KOKKOS_INLINE_FUNCTION double twin() const {
    double s = 0.0;
    for (int g = 0; g < 5; ++g) {
      const int h = set_of[g];
      s += ld::student_t1_distrib{x[g], theta[h], theta[2 + h]}.logpdf(y[g]);
    }
    return s;
  }
};

// the heterogeneous product: enet x 2 shared, mvgaussian2 (d = 2) x 2 by copy, pseudohuber x 1;
// theta = (lambda, alpha, 3 for copy 0 of Q, 3 for copy 1, sigma, nu)
struct mixed_case {
  double theta[10] = {1.3, 0.4, 0.3, -0.2, 0.4, -0.1, 0.2, 0.1, 0.9, 1.6};
  double y[7] = {0.7, -1.1, 0.4, 0.9, -0.2, 1.2, 0.5};
  double x[7] = {0.2, 0.3, -0.6, 0.1, 0.3, 0.4, -0.3};
  KOKKOS_INLINE_FUNCTION mixed_t make() const {
    const ld::mvgaussian2_distrib<log_chol> g{log_chol(2)};
    return mixed_t(ld::independent_product_component<uam<ld::enet_distrib>>{{}, 2, 1, nullptr},
                   ld::independent_product_component<ld::mvgaussian2_distrib<log_chol>>{g, 2, 2, nullptr},
                   ld::independent_product_component<uam<ld::pseudohuber_distrib>>{{}, 1, 1, nullptr});
  }
  // w: scratch for the Gaussian copies
  KOKKOS_INLINE_FUNCTION double twin(double* w) const {
    double s = ld::enet_distrib{x[0], theta[0], theta[1]}.logpdf(y[0]) +
               ld::enet_distrib{x[1], theta[0], theta[1]}.logpdf(y[1]) +
               ld::pseudohuber_distrib{x[6], theta[8], theta[9]}.logpdf(y[6]);
    const ld::mvgaussian2_distrib<log_chol> g{log_chol(2)};
    for (int c = 0; c < 2; ++c) {
      const double* th = theta + 2 + 3 * c;  // param-major: the two copies of Q side by side
      const auto st = g.theta_state(th, w);
      s += g.logpdf(st, y + 2 + 2 * c, x + 2 + 2 * c);
    }
    return s;
  }
};

constexpr int work_size = 4096;

void run_kernel(Kokkos::View<double*> res, Kokkos::View<double*> work) {
  Kokkos::parallel_for(
      "independent_product", 1, KOKKOS_LAMBDA(int) {
        double* w = &work(0);
        const student_case s;
        all_checks(s.make(), s.theta, s.y, s.x, s.twin(), w, &res(0));
        const mixed_case m;
        const double t = m.twin(w);
        all_checks(m.make(), m.theta, m.y, m.x, t, w, &res(n_checks));
      });
}

void check_results(const double* res) {
  CHECK(res[0] < 1e-14);
  for (int k = 1; k < n_checks; ++k) {
    CAPTURE(k);
    CHECK(res[k] < 1e-6);
  }
}

// the dense route equals the block route on the diagonal blocks and is zero elsewhere:
// J = 2 (packed over d) and J = 1, A = 1 (d x d), at the tuple idx
template <class P, int K>
void check_dense_against_blocks(const P& p, const double* th, const double* y, const double* x, const int* idx) {
  std::vector<double> w(p.state_size() + p.scratch_size() + work_size);
  const auto st = p.theta_state(th, w.data());
  double* ws = w.data() + p.state_size();
  const int d = p.dim();
  std::vector<double> dn(d * d), bl(d * d);
  REQUIRE(p.template logpdf_dy_dk<2, K>(st, y, x, idx, dn.data(), ws));
  REQUIRE(ld::mv_logpdf_dy_dk_blocks<2, K>(p, st, y, x, idx, bl.data(), ws));
  std::vector<double> dm(d * d), bm(d * d);
  REQUIRE(p.template logpdf_dy_dk<1, K, 1>(st, y, x, idx, dm.data(), ws));
  REQUIRE(ld::mv_logpdf_dy_dk_blocks<1, K, 1>(p, st, y, x, idx, bm.data(), ws));
  int n_packed = 0;
  for (int g = 0; g < p.n_blocks(); ++g) {
    const int m = p.block_start(ld::y_slot, g + 1) - p.block_start(ld::y_slot, g);
    n_packed += m * (m + 1) / 2;
  }
  CHECK(ld::mv_block_tensor_size(p, 2) == n_packed);
  int o2 = 0, o1 = 0;
  for (int g = 0; g < p.n_blocks(); ++g) {
    const int s = p.block_start(ld::y_slot, g), m = p.block_start(ld::y_slot, g + 1) - s;
    for (int i = s; i < s + m; ++i)
      for (int j = s; j < s + m; ++j) {
        if (j <= i) {
          CHECK(dn[i * (i + 1) / 2 + j] == bl[o2 + (i - s) * (i - s + 1) / 2 + (j - s)]);
          dn[i * (i + 1) / 2 + j] = 0.0;  // seen
        }
        CHECK(dm[i * d + j] == bm[o1 + (i - s) * m + (j - s)]);
        dm[i * d + j] = 0.0;
      }
    o2 += m * (m + 1) / 2;
    o1 += m * m;
  }
  for (int i = 0; i < d * (d + 1) / 2; ++i) CHECK(dn[i] == 0.0);
  for (int i = 0; i < d * d; ++i) CHECK(dm[i] == 0.0);
}

}  // namespace product_test

TEST_CASE("univariate_as_multivariate: the components of the univariate tensors") {
  using namespace product_test;
  const uam<ld::student_t2_distrib> m;
  const double th[2] = {1.3, 4.5}, y[1] = {0.7}, x[1] = {-0.2};
  const ld::student_t2_distrib d{-0.2, 1.3, 4.5};
  const auto st = m.theta_state(th, nullptr);
  CHECK(m.logpdf(st, y, x) == d.logpdf(0.7));
  CHECK(m.dim() == 1);
  CHECK(m.theta_size() == 2);
  CHECK(m.param_offset(2) == 1);
  double o;
  const int i01[2] = {0, 1}, i11[2] = {1, 1};
  REQUIRE(m.logpdf_dk<1>(st, y, x, i11, &o, nullptr));
  CHECK(o == d.logpdf_dy_dk<0, 1>(0.7)[2]);
  REQUIRE(m.logpdf_dk<0, 2>(st, y, x, nullptr, &o, nullptr));
  CHECK(o == d.logpdf_dy_dk<0, 2>(0.7)[0]);
  REQUIRE(m.logpdf_dy_dk<2, 2, 1>(st, y, x, i01, &o, nullptr));
  CHECK(o == d.logpdf_dy_dk<2, 3>(0.7)[ld::sym_tensor_position<3, 3>({0, 1, 2})]);
  REQUIRE(m.logpdf_dy_dk<1, 2>(st, y, x, i11, &o, nullptr));
  CHECK(o == d.logpdf_dy_dk<1, 2>(0.7)[ld::sym_tensor_position<3, 2>({2, 2})]);
}

TEST_CASE("independent_product_distrib: twin and finite differences, in a kernel and on the host") {
  using namespace product_test;
  Kokkos::View<double*> res("res", 2 * n_checks), work("work", work_size);
  run_kernel(res, work);
  auto h = Kokkos::create_mirror_view(res);
  Kokkos::deep_copy(h, res);
  check_results(&h(0));
  check_results(&h(n_checks));

  std::vector<double> w(work_size);
  double r0[n_checks], r1[n_checks];
  const student_case s;
  all_checks(s.make(), s.theta, s.y, s.x, s.twin(), w.data(), r0);
  const mixed_case m;
  const double t = m.twin(w.data());
  all_checks(m.make(), m.theta, m.y, m.x, t, w.data(), r1);
  check_results(r0);
  check_results(r1);
  for (int k = 0; k < n_checks; ++k) {
    CAPTURE(k);
    CHECK(std::fabs(r0[k] - h(k)) < 1e-9);
    CHECK(std::fabs(r1[k] - h(n_checks + k)) < 1e-9);
  }
}

TEST_CASE("independent_product_distrib: sizes, blocks, the dense route and tuples across sets") {
  using namespace product_test;
  const mixed_case c;
  const mixed_t p = c.make();
  CHECK(p.dim() == 7);
  CHECK(p.n_blocks() == 5);
  CHECK(p.theta_size() == 10);
  CHECK(p.param_size(3) == 6);
  CHECK(p.param_offset(3) == 2);
  CHECK(p.param_offset(5) == 9);
  const int starts[6] = {0, 1, 2, 4, 6, 7};
  for (int g = 0; g <= 5; ++g) CHECK(p.block_start(ld::y_slot, g) == starts[g]);
  check_dense_against_blocks<mixed_t, 0>(p, c.theta, c.y, c.x, nullptr);
  const int same[2] = {5, 7}, lam[1] = {0}, across[2] = {4, 5}, mix[2] = {0, 8};
  check_dense_against_blocks<mixed_t, 2>(p, c.theta, c.y, c.x, same);
  check_dense_against_blocks<mixed_t, 1>(p, c.theta, c.y, c.x, lam);
  std::vector<double> w(p.state_size() + p.scratch_size() + work_size), out(64, 7.0);
  const auto st = p.theta_state(c.theta, w.data());
  double* ws = w.data() + p.state_size();
  CHECK(!p.logpdf_dy_dk_blocks<1, 2>(st, c.y, c.x, across, out.data(), ws));  // Q of copy 0 and of copy 1
  CHECK(!p.logpdf_dy_dk_blocks<0, 2>(st, c.y, c.x, mix, out.data(), ws));     // enet and pseudohuber
  CHECK(out[0] == 7.0);
}

TEST_CASE("independent_product penalties: twin, block derivatives, rho") {
  using namespace product_test;
  {  // lasso over 4 coefficients, shared scale, prior location mu_0
    const lasso_t p(ld::independent_product_component<uam<ld::laplace1_distrib>>{{}, 4, 1, nullptr});
    const double mu0[4] = {0.1, 0.0, -0.3, 0.2}, beta[4] = {0.7, -0.4, 0.5, 0.1}, theta[1] = {0.6};
    const lasso_pen pen(p, mu0);
    std::vector<double> w(pen.state_size() + pen.scratch_size());
    const auto st = pen.theta_state(theta, w.data());
    double* ws = w.data() + pen.state_size();
    double v = 0.0;
    for (int g = 0; g < 4; ++g) v -= ld::laplace1_distrib{mu0[g], theta[0]}.logpdf(beta[g]);
    CHECK(std::fabs(pen.value(st, beta, ws) - v) < 1e-14);
    double g1[4], g2[4];
    const int i0[1] = {0};
    REQUIRE(pen.value_dbeta_dk<1, 1>(st, beta, i0, g1, ws));
    for (int g = 0; g < 4; ++g) {
      g2[g] = ld::laplace1_distrib{mu0[g], theta[0]}.logpdf_dy_dk<1, 1>(beta[g])[1];
      CHECK(g1[g] == -g2[g]);
    }
    // order 0 is the scalar in the block layout: the derivative in theta of the whole penalty
    double s0 = 0.0, s1 = 0.0;
    CHECK(ld::mv_block_tensor_size(p, 0) == 1);
    REQUIRE(pen.value_dbeta_dk<0, 1>(st, beta, i0, &s0, ws));
    for (int g = 0; g < 4; ++g) s1 -= ld::laplace1_distrib{mu0[g], theta[0]}.logpdf_dy_dk<0, 1>(beta[g])[1];
    CHECK(std::fabs(s0 - s1) < 1e-14);
    double h2[4];
    REQUIRE(pen.value_dbeta_dk<2, 0>(st, beta, nullptr, h2, ws));  // the block layout: one entry per coordinate
    for (int g = 0; g < 4; ++g) CHECK(h2[g] == 0.0);
    CHECK(pen.weak_convexity(theta) == 0.0);
  }
  {  // Student by group: rho against the least pen'' on a grid, per set
    const student_case c;
    const student_pen pen(c.make());
    const double rho = pen.weak_convexity(c.theta);
    double rho_set[2];
    for (int h = 0; h < 2; ++h) {
      const double sigma = c.theta[h], nu = c.theta[2 + h];
      rho_set[h] = (nu + 1.0) / (8.0 * nu * sigma * sigma);
      const lpen::distrib_penalty<uam<ld::student_t1_distrib>> one{uam<ld::student_t1_distrib>{}};
      const double th1[2] = {sigma, nu};
      double w1[4];
      const auto st = one.theta_state(th1, w1);
      double least = 0.0, h2 = 0.0;
      for (double b = 0.0; b < 20.0 * sigma; b += 1e-4 * sigma) {
        REQUIRE(one.value_dbeta_dk<2, 0>(st, &b, nullptr, &h2, w1));
        least = std::min(least, h2);
      }
      CHECK(std::fabs(-least - rho_set[h]) < 1e-7 * rho_set[h]);
    }
    CHECK(rho == std::max(rho_set[0], rho_set[1]));
    const lpen::distrib_penalty<uam<ld::student_t2_distrib>> t2{uam<ld::student_t2_distrib>{}};
    const double th2[2] = {1.5, 5.0};
    CHECK(std::fabs(t2.weak_convexity(th2) - 6.0 / (8.0 * 3.0 * 2.25)) < 1e-15);
  }
  {  // mixed: pen = -logpdf with a zero location, rho = 0 (all convex)
    const mixed_case c;
    const mixed_pen pen(c.make());
    std::vector<double> w(pen.state_size() + pen.scratch_size() + work_size);
    const auto st = pen.theta_state(c.theta, w.data());
    double* ws = w.data() + pen.state_size();
    const double zero[7] = {};
    mixed_case z = c;
    for (int i = 0; i < 7; ++i) z.x[i] = zero[i];
    const double v = pen.value(st, c.y, ws);
    CHECK(std::fabs(v + z.twin(ws)) < 1e-13);
    CHECK(pen.weak_convexity(c.theta) == 0.0);
  }
}

TEST_CASE("independent_product penalty: exp(-pen) integrates to one") {
  using namespace product_test;
  using P = ld::independent_product_distrib<uam<ld::pseudohuber_distrib>, uam<ld::gaussian2_distrib>>;
  const P p(ld::independent_product_component<uam<ld::pseudohuber_distrib>>{{}, 1, 1, nullptr},
            ld::independent_product_component<uam<ld::gaussian2_distrib>>{{}, 1, 1, nullptr});
  const double mu0[2] = {0.3, -0.2}, theta[3] = {0.7, 1.4, 0.5};
  const lpen::distrib_penalty<P> pen(p, mu0);
  std::vector<double> w(pen.state_size() + pen.scratch_size());
  const auto st = pen.theta_state(theta, w.data());
  double* ws = w.data() + pen.state_size();
  const double h = 0.02, L = 30.0;
  double sum = 0.0;
  for (double a = -L; a <= L; a += h)
    for (double b = -8.0; b <= 8.0; b += h) {
      const double beta[2] = {mu0[0] + a, mu0[1] + b};
      sum += std::exp(-pen.value(st, beta, ws));
    }
  CHECK(std::fabs(sum * h * h - 1.0) < 1e-9);
}
