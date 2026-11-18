// linear_map_penalty<P, Map>, penalties step 4 (note/penalties.md section 5d):
//  - the descriptors (class, kinks, structure, prox) and the maps (D, D^T, the
//    bound on ||D||_2^2);
//  - the Gaussian on differences of order k = 1, 2, 3 against the intrinsic
//    mvgaussian2 with Q = lambda D^T D (value with log det(D D^T), derivatives
//    in beta to order 3 in the dense layout, in theta by d/dv = -v d/drho, rank
//    m), on the host, since weighted_sum_psdmat is host only;
//  - in a kernel, the diagonal map against the product with the scales moved
//    onto beta: gaussian2 with v_j = v / w_j^2, laplace1 with sigma_j =
//    sigma / w_j and mu_j = mu0_j / w_j (exact, constant included): value,
//    derivatives, prox (the step per coordinate);
//  - in a kernel, Condat's prox of the fused lasso by its exact certificate
//    (the dual variable z_i = sum_{j <= i} (v_j - x_j), |z_i| <= t a, z_i =
//    -t a sign((Dx)_i - mu0_i) on a jump, z_{p-1} = 0), p from 2 to 1e4, t
//    from 1e-10 to 1e8, with and without mu0; NaN when the rates differ;
//  - on the host, Condat against the dual solved by FISTA (gap 1e-13);
//  - in a kernel, the Student on second differences: derivatives in beta to
//    order 3 against central differences of the order below;
//  - in a kernel, c(D) = -1/2 log det(D D^T) against mpmath (definition route,
//    tools/reference/penalty_difference_logdet.py);
//  - rho: the diagonal map's bound equals max_j(-pen'') on a grid.
#include <doctest/doctest.h>

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/pdmat/weighted_sum.hpp>
#include <laplace/penalty/linear_map_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace linear_map_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
namespace lpen = laplace::penalty;
template <class D>
using uam = ld::univariate_as_multivariate<D>;
template <class D>
using product = ld::independent_product_distrib<uam<D>>;
template <class D>
using component = ld::independent_product_component<uam<D>>;
template <class D>
using prod_pen = lpen::distrib_penalty<product<D>>;

using gauss_pen = prod_pen<ld::gaussian2_distrib>;
using lasso_pen = prod_pen<ld::laplace1_distrib>;
using lasso2_pen = prod_pen<ld::laplace2_distrib>;
using enet_pen = prod_pen<ld::enet_distrib>;
using student_pen = prod_pen<ld::student_t1_distrib>;
template <class P, int K>
using diff_pen = lpen::linear_map_penalty<P, lpen::difference_map<K>>;
template <class P>
using diag_pen = lpen::linear_map_penalty<P, lpen::diagonal_map>;

// descriptors
static_assert(lpen::penalty<diff_pen<lasso_pen, 1>> && lpen::penalty<diag_pen<lasso_pen>>);
static_assert(diff_pen<lasso_pen, 1>::normalization == lpen::penalty_normalization::intrinsic &&
              diag_pen<lasso_pen>::normalization == lpen::penalty_normalization::proper &&
              diff_pen<gauss_pen, 2>::normalization == lpen::penalty_normalization::intrinsic);
static_assert(diff_pen<lasso_pen, 1>::kinks == lpen::penalty_kink_geometry::linear_map &&
              diag_pen<lasso_pen>::kinks == lpen::penalty_kink_geometry::per_coordinate &&
              diff_pen<gauss_pen, 1>::kinks == lpen::penalty_kink_geometry::none);
static_assert(diff_pen<lasso_pen, 1>::structure == lpen::penalty_structure::coupled &&
              diag_pen<lasso_pen>::structure == lpen::penalty_structure::separable);
static_assert(diff_pen<lasso_pen, 1>::prox == lpen::penalty_prox_kind::closed_form &&
              diff_pen<lasso2_pen, 1>::prox == lpen::penalty_prox_kind::closed_form &&
              diag_pen<lasso_pen>::prox == lpen::penalty_prox_kind::closed_form &&
              diag_pen<enet_pen>::prox == lpen::penalty_prox_kind::closed_form);
static_assert(diff_pen<lasso_pen, 2>::prox == lpen::penalty_prox_kind::none &&
              diff_pen<enet_pen, 1>::prox == lpen::penalty_prox_kind::none &&
              diag_pen<gauss_pen>::prox == lpen::penalty_prox_kind::none &&
              diff_pen<gauss_pen, 1>::prox == lpen::penalty_prox_kind::none);
static_assert(diag_pen<lasso_pen>::scale_mixture == lpen::penalty_scale_mixture::exponential_variance);

KOKKOS_INLINE_FUNCTION int tensor_size(int p, int J) { return ld::binomial_coef(p + J - 1, J); }

// ---------------------------------------------------------------------------
// The Gaussian on differences against the intrinsic mvgaussian2 (host)

using ws_t = lp::weighted_sum_psdmat;
using host_matrix = ws_t::host_matrix;
using twin_pen = lpen::distrib_penalty<ld::mvgaussian2_distrib<ws_t>>;

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

// relative difference of two blocks; a structurally zero block (false) reads as zeros
template <class A, class B, int J, int KK>
double block_gap(const A& a, const typename A::state_type& sa, const B& b, const typename B::state_type& sb,
                 const double* beta, int n, double factor, double* wa, double* wb) {
  std::vector<double> x(n, 0.0), y(n, 0.0);
  const int idx[1] = {0};
  if (!a.template value_dbeta_dk<J, KK>(sa, beta, idx, x.data(), wa)) x.assign(n, 0.0);
  if (!b.template value_dbeta_dk<J, KK>(sb, beta, idx, y.data(), wb)) y.assign(n, 0.0);
  double e = 0.0, s = 1.0;
  for (int k = 0; k < n; ++k) s = std::fmax(s, std::fabs(y[k]));
  for (int k = 0; k < n; ++k) e = std::fmax(e, std::fabs(factor * x[k] - y[k]));
  return e / s;
}

template <int K>
void gauss_twin(int p, double lambda, double* worst) {
  const int m = p - K;
  const gauss_pen inner(product<ld::gaussian2_distrib>(component<ld::gaussian2_distrib>{{}, m, 1, nullptr}));
  const diff_pen<gauss_pen, K> pen(inner, lpen::difference_map<K>(p));
  const ws_t s(std::vector<host_matrix>{gram_of_differences(p, K)});
  REQUIRE(s.rank() == m);
  const twin_pen twin{ld::mvgaussian2_distrib<ws_t>(s)};
  const double v = 1.0 / lambda, ta[1] = {v}, tb[1] = {std::log(lambda)};
  std::vector<double> wa(pen.state_size() + pen.scratch_size()), wb(twin.state_size() + twin.scratch_size());
  const auto sa = pen.theta_state(ta, wa.data());
  const auto sb = twin.theta_state(tb, wb.data());
  double* xa = wa.data() + pen.state_size();
  double* xb = wb.data() + twin.state_size();
  std::vector<double> beta(p);
  for (int j = 0; j < p; ++j) beta[j] = std::sin(1.7 * j + 0.3) + 0.1 * j;
  // d/drho = -v d/dv for rho = log lambda, v = 1 / lambda
  const double r[6] = {
      block_gap<decltype(pen), twin_pen, 0, 0>(pen, sa, twin, sb, beta.data(), 1, 1.0, xa, xb),
      block_gap<decltype(pen), twin_pen, 1, 0>(pen, sa, twin, sb, beta.data(), tensor_size(p, 1), 1.0, xa, xb),
      block_gap<decltype(pen), twin_pen, 2, 0>(pen, sa, twin, sb, beta.data(), tensor_size(p, 2), 1.0, xa, xb),
      block_gap<decltype(pen), twin_pen, 3, 0>(pen, sa, twin, sb, beta.data(), tensor_size(p, 3), 1.0, xa, xb),
      block_gap<decltype(pen), twin_pen, 0, 1>(pen, sa, twin, sb, beta.data(), 1, -v, xa, xb),
      block_gap<decltype(pen), twin_pen, 1, 1>(pen, sa, twin, sb, beta.data(), tensor_size(p, 1), -v, xa, xb)};
  for (int i = 0; i < 6; ++i) worst[i] = std::fmax(worst[i], r[i]);
}

// ---------------------------------------------------------------------------
// The diagonal map against the scales moved onto beta (kernel)

constexpr int pd = 5;

// res[0] gaussian value and derivatives; res[1] laplace1 value and derivatives; res[2] laplace1 prox
KOKKOS_INLINE_FUNCTION void diagonal_twins(double* res) {
  const double wt[pd] = {0.5, 1.0, 2.0, 3.5, 0.125};
  const double beta[pd] = {0.4, -1.3, 0.05, 2.2, -0.7};
  const double mu0[pd] = {0.3, 0.0, -0.4, 1.1, 0.2};
  const lpen::diagonal_map map(wt, pd);
  double w[64], w2[64];
  res[0] = res[1] = res[2] = 0.0;
  {  // gaussian2 with v on W beta against gaussian2 with v / w_j^2 on beta
    const double v = 0.8, th[1] = {v};
    double th2[pd];
    for (int j = 0; j < pd; ++j) th2[j] = v / (wt[j] * wt[j]);
    const diag_pen<gauss_pen> a(
        gauss_pen(product<ld::gaussian2_distrib>(component<ld::gaussian2_distrib>{{}, pd, 1, nullptr})), map);
    const gauss_pen b(product<ld::gaussian2_distrib>(component<ld::gaussian2_distrib>{{}, pd, pd, nullptr}));
    const auto sa = a.theta_state(th, w);
    const auto sb = b.theta_state(th2, w2);
    double* xa = w + a.state_size();
    double* xb = w2 + b.state_size();
    double oa[pd], ob[pd];
    const int idx[1] = {0};
    res[0] = Kokkos::fabs(a.value(sa, beta, xa) - b.value(sb, beta, xb));
    a.template value_dbeta_dk<1, 0>(sa, beta, idx, oa, xa);
    b.template value_dbeta_dk<1, 0>(sb, beta, idx, ob, xb);
    for (int j = 0; j < pd; ++j) res[0] = Kokkos::fmax(res[0], Kokkos::fabs(oa[j] - ob[j]));
    a.template value_dbeta_dk<2, 0>(sa, beta, idx, oa, xa);
    b.template value_dbeta_dk<2, 0>(sb, beta, idx, ob, xb);
    for (int j = 0; j < pd; ++j) res[0] = Kokkos::fmax(res[0], Kokkos::fabs(oa[j] - ob[j]) / Kokkos::fabs(ob[j]));
  }
  {  // laplace1 with sigma and mu0 on W beta against laplace1 with sigma / w_j and mu0_j / w_j
    const double sigma = 1.3, th[1] = {sigma};
    double th2[pd], m2[pd];
    for (int j = 0; j < pd; ++j) {
      th2[j] = sigma / wt[j];
      m2[j] = mu0[j] / wt[j];
    }
    const diag_pen<lasso_pen> a(
        lasso_pen(product<ld::laplace1_distrib>(component<ld::laplace1_distrib>{{}, pd, 1, nullptr}), mu0), map);
    const lasso_pen b(product<ld::laplace1_distrib>(component<ld::laplace1_distrib>{{}, pd, pd, nullptr}), m2);
    const auto sa = a.theta_state(th, w);
    const auto sb = b.theta_state(th2, w2);
    double* xa = w + a.state_size();
    double* xb = w2 + b.state_size();
    double oa[pd], ob[pd];
    const int idx[1] = {0};
    res[1] = Kokkos::fabs(a.value(sa, beta, xa) - b.value(sb, beta, xb));
    a.template value_dbeta_dk<1, 0>(sa, beta, idx, oa, xa);
    b.template value_dbeta_dk<1, 0>(sb, beta, idx, ob, xb);
    for (int j = 0; j < pd; ++j) res[1] = Kokkos::fmax(res[1], Kokkos::fabs(oa[j] - ob[j]));
    a.template value_dbeta_dk<0, 1>(sa, beta, idx, oa, xa);
    const double d_sigma = oa[0];
    // the twin's d/dsigma_j sums, through sigma_j = sigma / w_j, to d/dsigma
    double s = 0.0;
    for (int j = 0; j < pd; ++j) {
      const int ij[1] = {j};
      b.template value_dbeta_dk<0, 1>(sb, beta, ij, ob, xb);
      s += ob[0] / wt[j];
    }
    res[1] = Kokkos::fmax(res[1], Kokkos::fabs(d_sigma - s));
    const double ts[4] = {1e-6, 0.2, 1.7, 40.0};
    const double v[pd] = {0.9, -0.05, -2.0, 0.31, 6.0};
    for (int i = 0; i < 4; ++i) {
      a.prox_beta(sa, v, ts[i], oa, xa);
      b.prox_beta(sb, v, ts[i], ob, xb);
      for (int j = 0; j < pd; ++j)
        res[2] = Kokkos::fmax(res[2], Kokkos::fabs(oa[j] - ob[j]) / Kokkos::fmax(1.0, Kokkos::fabs(ob[j])));
    }
  }
}

// ---------------------------------------------------------------------------
// Condat's prox by its certificate (kernel)

// a step signal with noise from a linear congruential generator, length p
KOKKOS_INLINE_FUNCTION void signal(int p, unsigned seed, double* v) {
  unsigned s = seed;
  for (int j = 0; j < p; ++j) {
    s = 1664525u * s + 1013904223u;
    const double noise = double(s >> 8) / double(1u << 24) - 0.5;
    v[j] = double((j * 7) / (p > 6 ? p / 6 + 1 : 1) % 3) - 1.0 + 0.8 * noise;
  }
}

// the certificate's violation, relative to eps-scale sums; res[0] the bound
// |z_i| <= t a, res[1] the jumps z_i = -t a sign, res[2] z_{p-1} = 0
KOKKOS_INLINE_FUNCTION void certify(int p, const double* v, const double* x, const double* mu0, double lam,
                                    double* res) {
  double z = 0.0, scale = 0.0, u = 0.0;
  for (int i = 0; i < p; ++i) {
    z += v[i] - x[i];
    scale += Kokkos::fabs(v[i]) + Kokkos::fabs(x[i]);
    if (i == p - 1) break;
    const double d = (x[i + 1] - x[i]) - (mu0 != nullptr ? mu0[i] : 0.0);
    const double tol = 1e-12 * (Kokkos::fabs(x[i + 1]) + Kokkos::fabs(x[i]) + 1.0);
    const double denom = scale + lam;
    res[0] = Kokkos::fmax(res[0], Kokkos::fmax(0.0, Kokkos::fabs(z) - lam) / denom);
    if (Kokkos::fabs(d) > tol) res[1] = Kokkos::fmax(res[1], Kokkos::fabs(z + lam * (d > 0 ? 1.0 : -1.0)) / denom);
    u = denom;
  }
  res[2] = Kokkos::fmax(res[2], Kokkos::fabs(z) / (p > 1 ? u : scale + 1.0));
}

constexpr int n_sizes = 7;
constexpr int max_p = 10000;

void condat_kernel(double* host) {
  Kokkos::View<double[6]> res("condat_res");
  Kokkos::View<double*> buf("condat_buf", 8 * max_p + 64);
  Kokkos::parallel_for(
      "condat_certificate", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        const int sizes[n_sizes] = {2, 3, 5, 17, 100, 1000, max_p};
        const double ts[5] = {1e-10, 0.05, 0.5, 3.0, 1e8};
        double r[6] = {0, 0, 0, 0, 0, 0};
        double* v = buf.data();
        double* x = v + max_p;
        double* mu0 = x + max_p;
        double* w = mu0 + max_p;
        for (int si = 0; si < n_sizes; ++si) {
          const int p = sizes[si];
          signal(p, 17u + si, v);
          for (int i = 0; i + 1 < p; ++i) mu0[i] = 0.02 * ((i % 5) - 2);
          for (int located = 0; located < 2; ++located) {
            const double sigma = 0.7, th[1] = {sigma};
            const lasso_pen inner(product<ld::laplace1_distrib>(component<ld::laplace1_distrib>{{}, p - 1, 1, nullptr}),
                                  located ? mu0 : nullptr);
            const diff_pen<lasso_pen, 1> pen(inner, lpen::difference_map<1>(p));
            const auto st = pen.theta_state(th, w);
            double* ws = w + pen.state_size();
            for (int ti = 0; ti < 5; ++ti) {
              const double t = ts[ti];
              pen.prox_beta(st, v, t, x, ws);
              certify(p, v, x, located ? mu0 : nullptr, t / sigma, r);
              if (ti == 4 && !located) {  // a huge step: the constant at the mean of v
                double mean = 0.0;
                for (int i = 0; i < p; ++i) mean += v[i];
                mean /= p;
                for (int i = 0; i < p; ++i) r[3] = Kokkos::fmax(r[3], Kokkos::fabs(x[i] - mean));
              }
              if (ti == 0)  // a tiny step: the identity within t a on each coordinate
                for (int i = 0; i < p; ++i) r[4] = Kokkos::fmax(r[4], Kokkos::fabs(x[i] - v[i]) / (2.0 * t / sigma));
            }
          }
        }
        {  // two rates that differ: NaN out
          const int groups[3] = {0, 1, 0};
          const double th[2] = {0.7, 0.9};
          const lasso_pen inner(product<ld::laplace1_distrib>(component<ld::laplace1_distrib>{{}, 3, 2, groups}));
          const diff_pen<lasso_pen, 1> pen(inner, lpen::difference_map<1>(4));
          const auto st = pen.theta_state(th, w);
          pen.prox_beta(st, v, 0.3, x, w + pen.state_size());
          r[5] = (Kokkos::isnan(x[0]) && Kokkos::isnan(x[3])) ? 0.0 : 1.0;
        }
        for (int i = 0; i < 6; ++i) res(i) = r[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 6; ++i) host[i] = h(i);
}

// the dual of the TV prox by FISTA on the host: x = y - D^T z, |z|_inf <= lam
// dual value (1/2)|y|^2 - (1/2)|x|^2; stop when the gap to the primal is below gap max(1, primal)
std::vector<double> tv_dual_fista(const std::vector<double>& y, double lam, double gap) {
  const int p = static_cast<int>(y.size()), m = p - 1;
  std::vector<double> z(m, 0.0), q(m, 0.0), x(p), zn(m);
  auto primal_of = [&](const std::vector<double>& zz) {
    for (int j = 0; j < p; ++j) x[j] = y[j] - ((j > 0 ? zz[j - 1] : 0.0) - (j < m ? zz[j] : 0.0));
  };
  double s = 1.0;
  for (int it = 0; it < 2000000; ++it) {
    primal_of(q);
    for (int i = 0; i < m; ++i) {
      // gradient of 1/2 |D^T z - y|^2 is D (D^T z - y) = -D x
      const double g = -(x[i + 1] - x[i]);
      zn[i] = std::fmin(lam, std::fmax(-lam, q[i] - g / 4.0));
    }
    const double sn = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * s * s));
    for (int i = 0; i < m; ++i) q[i] = zn[i] + (s - 1.0) / sn * (zn[i] - z[i]);
    z = zn;
    s = sn;
    primal_of(z);
    double pr = 0.0, du = 0.0, yy = 0.0;
    for (int j = 0; j < p; ++j) {
      pr += 0.5 * (x[j] - y[j]) * (x[j] - y[j]);
      du += 0.5 * x[j] * x[j];
      yy += 0.5 * y[j] * y[j];
    }
    for (int i = 0; i < m; ++i) pr += lam * std::fabs(x[i + 1] - x[i]);
    if (pr - (yy - du) <= gap * std::fmax(1.0, pr)) break;
  }
  return x;
}

// ---------------------------------------------------------------------------
// The Student on second differences: derivatives by central differences (kernel)

constexpr int ps = 6;

template <int J>
KOKKOS_INLINE_FUNCTION void student_block(const diff_pen<student_pen, 2>& pen, const double* th, const double* beta,
                                          double* out, double* w) {
  const auto st = pen.theta_state(th, w);
  const int idx[1] = {0};
  pen.template value_dbeta_dk<J, 0>(st, beta, idx, out, w + pen.state_size());
}

// worst relative error of the order-J block against differences of the order J - 1
template <int J>
KOKKOS_INLINE_FUNCTION double student_fd(const diff_pen<student_pen, 2>& pen, const double* th, const double* beta,
                                         double* w) {
  constexpr int n = J == 1 ? ps : (J == 2 ? ps * (ps + 1) / 2 : 56);
  constexpr int nl = J == 1 ? 1 : (J == 2 ? ps : ps * (ps + 1) / 2);
  double exact[n], lo[nl], hi[nl], b[ps];
  student_block<J>(pen, th, beta, exact, w);
  double e = 0.0, s = 0.0;
  for (int k = 0; k < n; ++k) s = Kokkos::fmax(s, Kokkos::fabs(exact[k]));
  const double h = 1e-5;
  int t[J];
  for (int r = 0; r < J; ++r) t[r] = 0;
  do {  // the component at sorted tuple t: d/dbeta_{t0} of the order J - 1 component at t[1..]
    for (int j = 0; j < ps; ++j) b[j] = beta[j];
    b[t[0]] = beta[t[0]] + h;
    student_block<J - 1>(pen, th, b, hi, w);
    b[t[0]] = beta[t[0]] - h;
    student_block<J - 1>(pen, th, b, lo, w);
    const int pl = J == 1 ? 0 : ld::detail_mv::block_position(ps, J - 1, t + 1);
    const double fd = (hi[pl] - lo[pl]) / (2.0 * h);
    e = Kokkos::fmax(e, Kokkos::fabs(fd - exact[ld::detail_mv::block_position(ps, J, t)]));
  } while (ld::detail_mv::next_block_tuple(ps, J, t));
  return e / s;
}

void student_kernel(double* host) {
  Kokkos::View<double[3]> res("student_res");
  Kokkos::parallel_for(
      "student_differences", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        const double mu0[ps - 2] = {0.1, -0.2, 0.0, 0.3};
        const diff_pen<student_pen, 2> pen(
            student_pen(product<ld::student_t1_distrib>(component<ld::student_t1_distrib>{{}, ps - 2, 1, nullptr}), mu0),
            lpen::difference_map<2>(ps));
        const double th[2] = {0.6, 3.0};
        const double beta[ps] = {0.3, -0.4, 1.2, 0.5, -0.9, 0.2};
        double w[256];
        res(0) = student_fd<1>(pen, th, beta, w);
        res(1) = student_fd<2>(pen, th, beta, w);
        res(2) = student_fd<3>(pen, th, beta, w);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 3; ++i) host[i] = h(i);
}

// ---------------------------------------------------------------------------
// c(D) against mpmath (kernel)

KOKKOS_INLINE_FUNCTION double log_det_of(int k, int p) {
  switch (k) {
    case 1: return -2.0 * lpen::difference_map<1>(p).log_normalizer();
    case 2: return -2.0 * lpen::difference_map<2>(p).log_normalizer();
    case 3: return -2.0 * lpen::difference_map<3>(p).log_normalizer();
    default: return -2.0 * lpen::difference_map<4>(p).log_normalizer();
  }
}

double logdet_kernel(const std::vector<int>& ks, const std::vector<int>& ps_, const std::vector<double>& ref) {
  const int n = static_cast<int>(ks.size());
  Kokkos::View<int*> k("k", n), p("p", n);
  Kokkos::View<double*> r("r", n), e("e", n);
  auto hk = Kokkos::create_mirror_view(k);
  auto hp = Kokkos::create_mirror_view(p);
  auto hr = Kokkos::create_mirror_view(r);
  for (int i = 0; i < n; ++i) {
    hk(i) = ks[i];
    hp(i) = ps_[i];
    hr(i) = ref[i];
  }
  Kokkos::deep_copy(k, hk);
  Kokkos::deep_copy(p, hp);
  Kokkos::deep_copy(r, hr);
  Kokkos::parallel_for(
      "difference_log_det", n, KOKKOS_LAMBDA(int i) { e(i) = Kokkos::fabs(log_det_of(k(i), p(i)) - r(i)) / r(i); });
  auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) worst = std::fmax(worst, he(i));
  return worst;
}

}  // namespace linear_map_test

TEST_CASE("linear_map_penalty: the maps") {
  using namespace linear_map_test;
  const double wt[3] = {0.5, 2.0, 1.5};
  const lpen::diagonal_map w(wt, 3);
  CHECK(w.norm_squared_bound() == 4.0);
  CHECK(std::fabs(w.log_normalizer() + std::log(0.5 * 2.0 * 1.5)) < 1e-15);
  const lpen::difference_map<2> d(6);
  CHECK(d.m() == 4);
  CHECK(d.norm_squared_bound() == 16.0);
  // <D beta, g> = <beta, D^T g>
  const double beta[6] = {0.3, -1.0, 2.0, 0.7, 0.1, -0.4}, g[4] = {1.0, -0.5, 0.25, 2.0};
  double u[4], dt[6];
  d.apply(beta, u);
  d.apply_transpose(g, dt);
  double a = 0.0, b = 0.0;
  for (int i = 0; i < 4; ++i) a += u[i] * g[i];
  for (int j = 0; j < 6; ++j) b += beta[j] * dt[j];
  CHECK(std::fabs(a - b) < 1e-14);
  CHECK(u[0] == beta[2] - 2.0 * beta[1] + beta[0]);
}

TEST_CASE("linear_map_penalty: Gaussian on differences against the intrinsic mvgaussian2, on the host") {
  using namespace linear_map_test;
  double worst[6] = {0, 0, 0, 0, 0, 0};
  for (const double lambda : {0.3, 2.0}) {
    for (const int p : {3, 8, 12}) gauss_twin<1>(p, lambda, worst);
    for (const int p : {4, 8, 12}) gauss_twin<2>(p, lambda, worst);
    for (const int p : {5, 8, 12}) gauss_twin<3>(p, lambda, worst);
  }
  MESSAGE("value ", worst[0], ", J=1 ", worst[1], ", J=2 ", worst[2], ", J=3 ", worst[3], ", K=1 ", worst[4],
          ", J=1 K=1 ", worst[5]);
  for (int i = 0; i < 6; ++i) CHECK(worst[i] < 1e-14);
}

TEST_CASE("linear_map_penalty: diagonal map against the scales moved onto beta, in a kernel") {
  using namespace linear_map_test;
  Kokkos::View<double[3]> res("diag_res");
  Kokkos::parallel_for(
      "diagonal_twins", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        double r[3];
        diagonal_twins(r);
        for (int i = 0; i < 3; ++i) res(i) = r[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  MESSAGE("gaussian ", h(0), ", laplace1 ", h(1), ", prox ", h(2));
  CHECK(h(0) < 5e-14);
  CHECK(h(1) < 1e-14);
  CHECK(h(2) < 1e-15);
}

TEST_CASE("linear_map_penalty: Condat's prox by its certificate, in a kernel") {
  using namespace linear_map_test;
  double r[6];
  condat_kernel(r);
  MESSAGE("bound ", r[0], ", jumps ", r[1], ", closing ", r[2], ", huge t ", r[3], ", tiny t ", r[4]);
  CHECK(r[0] < 1e-15);
  CHECK(r[1] < 1e-15);
  CHECK(r[2] < 5e-14);
  CHECK(r[3] < 1e-7);
  CHECK(r[4] < 1.0 + 1e-5);
  CHECK(r[5] == 0.0);
}

TEST_CASE("linear_map_penalty: Condat against the dual solved by FISTA, on the host") {
  using namespace linear_map_test;
  double worst = 0.0;
  for (const int p : {10, 100}) {
    std::vector<double> v(p), x(p);
    signal(p, 5u, v.data());
    const double sigma = 0.5, th[1] = {sigma};
    const lasso_pen inner(product<ld::laplace1_distrib>(component<ld::laplace1_distrib>{{}, p - 1, 1, nullptr}));
    const diff_pen<lasso_pen, 1> pen(inner, lpen::difference_map<1>(p));
    std::vector<double> w(pen.state_size() + pen.scratch_size());
    const auto st = pen.theta_state(th, w.data());
    for (const double t : {0.1, 1.0}) {
      pen.prox_beta(st, v.data(), t, x.data(), w.data() + pen.state_size());
      const auto y = tv_dual_fista(v, t / sigma, 1e-13);
      for (int j = 0; j < p; ++j) worst = std::fmax(worst, std::fabs(x[j] - y[j]));
    }
  }
  MESSAGE("max |Condat - dual FISTA| ", worst);
  CHECK(worst < 1e-11);
}

TEST_CASE("linear_map_penalty: Student on second differences, derivatives by central differences, in a kernel") {
  using namespace linear_map_test;
  double r[3];
  student_kernel(r);
  MESSAGE("J=1 ", r[0], ", J=2 ", r[1], ", J=3 ", r[2]);
  for (int i = 0; i < 3; ++i) CHECK(r[i] < 5e-9);
}

TEST_CASE("linear_map_penalty: log det(D D^T) against mpmath, in a kernel") {
  using namespace linear_map_test;
  std::ifstream f(std::string(LAPLACE_REFERENCE_DIR) + "/penalty_difference_logdet.csv");
  REQUIRE(f.good());
  std::string line;
  std::getline(f, line);
  std::vector<int> ks, ps_;
  std::vector<double> ref;
  while (std::getline(f, line)) {
    std::stringstream s(line);
    std::string a, b, hi, lo;
    std::getline(s, a, ',');
    std::getline(s, b, ',');
    std::getline(s, hi, ',');
    std::getline(s, lo, ',');
    ks.push_back(std::stoi(a));
    ps_.push_back(std::stoi(b));
    ref.push_back(std::strtod(hi.c_str(), nullptr) + std::strtod(lo.c_str(), nullptr));
  }
  REQUIRE(ks.size() == 118);
  const double e = logdet_kernel(ks, ps_, ref);
  MESSAGE("log det(D D^T): worst relative error ", e);
  CHECK(e < 1e-15);
}

TEST_CASE("linear_map_penalty: rho of the diagonal map equals max(-pen'') on a grid") {
  using namespace linear_map_test;
  const double wt[2] = {0.5, 3.0};
  const double sigma = 0.8, nu = 4.0, th[2] = {sigma, nu};
  const diag_pen<student_pen> pen(
      student_pen(product<ld::student_t1_distrib>(component<ld::student_t1_distrib>{{}, 2, 1, nullptr})),
      lpen::diagonal_map(wt, 2));
  std::vector<double> w(pen.state_size() + pen.scratch_size());
  const auto st = pen.theta_state(th, w.data());
  const int idx[1] = {0};
  double m = 0.0, h[2];
  for (double z = -10.0; z <= 10.0; z += 1e-4) {
    const double beta[2] = {z, z / 6.0};
    pen.value_dbeta_dk<2, 0>(st, beta, idx, h, w.data() + pen.state_size());
    m = std::fmax(m, std::fmax(-h[0], -h[1]));
  }
  const double rho = pen.weak_convexity(th);
  MESSAGE("rho ", rho, ", max(-pen'') ", m);
  CHECK(std::fabs(rho - (nu + 1.0) / (8.0 * nu * sigma * sigma) * 9.0) < 1e-14);
  CHECK(std::fabs(m - rho) < 1e-7 * rho);
}
