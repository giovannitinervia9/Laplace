// The distribution function members of mvgaussian1_distrib and
// mvgaussian2_distrib (distrib/detail/mvgaussian_cdf.hpp): the values against
// laplace::math::log_mvn_rect on the covariance of the state, every derivative
// block against central differences of the block of the order below (in the
// location and in theta, cdf, ccdf and rectangle, both forms), the derivatives
// in y against those in the location, the named concept, and the quiet NaN of
// the class with a singular precision. The bounds are those of a central
// difference of step 1e-5 (about 1e-10 relative), with a margin.
#include <doctest/doctest.h>

#include <laplace/distrib/mvgaussian1.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/pdmat/unstructured.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace mvgaussian_cdf_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using S = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using D1 = ld::mvgaussian1_distrib<S>;
using D2 = ld::mvgaussian2_distrib<S>;
using imp_t = ld::mvgaussian2_distrib<lp::weighted_sum_psdmat>;

static_assert(ld::mv_has_cdf<D1> && ld::mv_has_cdf<D2> && ld::mv_has_cdf<imp_t>);

constexpr double I = std::numeric_limits<double>::infinity();

enum { lower = 0, upper = 1, rect = 2 };

template <class D, int K, int A>
void eval(const D& g, int reg, const double* th, const double* y, const double* b, const double* mu, const int* idx,
          double* out) {
  std::vector<double> w(g.state_size() + g.scratch_size() + g.cdf_scratch_size(K, A));
  const auto st = g.theta_state(th, w.data());
  double* ws = w.data() + g.state_size();
  if (reg == lower) g.template log_cdf_dk<K, A>(st, y, mu, idx, out, ws);
  if (reg == upper) g.template log_ccdf_dk<K, A>(st, y, mu, idx, out, ws);
  if (reg == rect) g.template log_rect_prob_dk<K, A>(st, y, b, mu, idx, out, ws);
}

int pos(int d, int n, const int* t) { return n == 0 ? 0 : ld::detail_mv::block_position(d, n, t); }

// the order-A block in mu against d/dmu_i of the order-(A - 1) block
template <class D, int K, int A>
double diff_mu(const D& g, int reg, const double* th, const double* y, const double* b, double* mu, const int* idx) {
  const int d = g.dim();
  std::vector<double> o(ld::sym_tensor_size(d, A)), op(ld::sym_tensor_size(d, A - 1)), om(op.size());
  eval<D, K, A>(g, reg, th, y, b, mu, idx, o.data());
  const double h = 1e-5;
  double worst = 0.0;
  for (int i = 0; i < d; ++i) {
    mu[i] += h;
    eval<D, K, A - 1>(g, reg, th, y, b, mu, idx, op.data());
    mu[i] -= 2 * h;
    eval<D, K, A - 1>(g, reg, th, y, b, mu, idx, om.data());
    mu[i] += h;
    int t[8] = {0}, t2[8];
    do {
      int m = 0;
      bool ins = false;
      for (int q = 0; q < A - 1; ++q) {
        if (!ins && i < t[q]) t2[m++] = i, ins = true;
        t2[m++] = t[q];
      }
      if (!ins) t2[m++] = i;
      const double fd = (op[pos(d, A - 1, t)] - om[pos(d, A - 1, t)]) / (2 * h);
      const double an = o[pos(d, A, t2)];
      worst = std::fmax(worst, std::fabs(fd - an) / std::fmax(1.0, std::fabs(an)));
    } while (A - 1 > 0 && ld::detail_mv::next_block_tuple(d, A - 1, t));
  }
  return worst;
}

// the order-K block in theta against d/dtheta_(idx[K-1]) of the order K - 1 block
template <class D, int K, int A>
double diff_theta(const D& g, int reg, double* th, const double* y, const double* b, const double* mu, const int* idx) {
  const int d = g.dim();
  std::vector<double> o(ld::sym_tensor_size(d, A)), op(o.size()), om(o.size());
  eval<D, K, A>(g, reg, th, y, b, mu, idx, o.data());
  const double h = 1e-5;
  const int j = idx[K - 1];
  th[j] += h;
  eval<D, K - 1, A>(g, reg, th, y, b, mu, idx, op.data());
  th[j] -= 2 * h;
  eval<D, K - 1, A>(g, reg, th, y, b, mu, idx, om.data());
  th[j] += h;
  double worst = 0.0;
  for (std::size_t q = 0; q < o.size(); ++q)
    worst = std::fmax(worst, std::fabs((op[q] - om[q]) / (2 * h) - o[q]) / std::fmax(1.0, std::fabs(o[q])));
  return worst;
}

template <class D>
void check_law(const D& g, int reg) {
  double th[6] = {0.1, 0.3, -0.2, 0.4, 0.2, -0.1};
  double y[3] = {0.2, -0.4, 0.7}, b[3] = {0.9, 0.3, I}, mu[3] = {0.1, -0.1, 0.2};
  if (reg == rect) y[2] = -0.5, b[2] = 1.5;
  const int idx[6] = {0, 1, 1, 3, 4, 5};
  const double e[] = {diff_mu<D, 0, 1>(g, reg, th, y, b, mu, idx),   diff_mu<D, 0, 2>(g, reg, th, y, b, mu, idx),
                      diff_mu<D, 0, 3>(g, reg, th, y, b, mu, idx),   diff_mu<D, 0, 4>(g, reg, th, y, b, mu, idx),
                      diff_mu<D, 0, 6>(g, reg, th, y, b, mu, idx),   diff_mu<D, 1, 2>(g, reg, th, y, b, mu, idx),
                      diff_theta<D, 1, 0>(g, reg, th, y, b, mu, idx), diff_theta<D, 2, 0>(g, reg, th, y, b, mu, idx),
                      diff_theta<D, 3, 0>(g, reg, th, y, b, mu, idx), diff_theta<D, 1, 1>(g, reg, th, y, b, mu, idx),
                      diff_theta<D, 2, 2>(g, reg, th, y, b, mu, idx), diff_theta<D, 4, 0>(g, reg, th, y, b, mu, idx)};
  for (double v : e) {
    INFO("region ", reg, ": relative difference ", v);
    CHECK(v <= 1e-8);
  }
}

// the members inside a kernel: log_cdf_dk<1, 2> of the precision form and
// log_rect_prob_dk<0, 3> of the covariance form, one thread per row of shifted y
struct DeviceRows {
  D1 g1;
  D2 g2;
  Kokkos::View<double**, Kokkos::LayoutRight> out, w;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const double th[6] = {0.1, 0.3, -0.2, 0.4, 0.2, -0.1};
    const double y[3] = {0.2 - 0.5 * i, -0.4, 0.7}, b[3] = {0.9, 0.3, 1.5}, a[3] = {-0.3 - 0.5 * i, -1.0, -0.5};
    const double mu[3] = {0.1, -0.1, 0.2};
    const int idx[1] = {3};
    double* ws = &w(i, 0);
    const auto st2 = g2.theta_state(th, ws);
    g2.template log_cdf_dk<1, 2>(st2, y, mu, idx, &out(i, 0), ws + g2.state_size());
    const auto st1 = g1.theta_state(th, ws);
    g1.template log_rect_prob_dk<0, 3>(st1, a, b, mu, idx, &out(i, 6), ws + g1.state_size());
  }
};

}  // namespace mvgaussian_cdf_test

using namespace mvgaussian_cdf_test;

// the kernel in a function with external linkage (note/brief_agenti.md)
Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> mvgaussian_cdf_device_rows(int n) {
  const S s(3);
  const D1 g1(s);
  const D2 g2(s);
  const int sc = g1.state_size() + g1.scratch_size() +
                 (g1.cdf_scratch_size(1, 2) > g1.cdf_scratch_size(0, 3) ? g1.cdf_scratch_size(1, 2) : g1.cdf_scratch_size(0, 3));
  Kokkos::View<double**, Kokkos::LayoutRight> out("out", n, 16), w("w", n, sc);
  Kokkos::parallel_for("mvgaussian_cdf", n, DeviceRows{g1, g2, out, w});
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

TEST_CASE("mvgaussian cdf: the members inside a kernel equal their host values") {
  const int n = 6;
  const auto dev = mvgaussian_cdf_device_rows(n);
  const S s(3);
  const D1 g1(s);
  const D2 g2(s);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double th[6] = {0.1, 0.3, -0.2, 0.4, 0.2, -0.1};
    const double y[3] = {0.2 - 0.5 * i, -0.4, 0.7}, b[3] = {0.9, 0.3, 1.5}, a[3] = {-0.3 - 0.5 * i, -1.0, -0.5};
    const double mu[3] = {0.1, -0.1, 0.2};
    const int idx[1] = {3};
    double h1[6], h2[10];
    eval<D2, 1, 2>(g2, lower, th, y, b, mu, idx, h1);
    eval<D1, 0, 3>(g1, rect, th, a, b, mu, idx, h2);
    for (int q = 0; q < 6; ++q) worst = std::fmax(worst, std::fabs(dev(i, q) - h1[q]) / std::fmax(1.0, std::fabs(h1[q])));
    for (int q = 0; q < 10; ++q) worst = std::fmax(worst, std::fabs(dev(i, 6 + q) - h2[q]) / std::fmax(1.0, std::fabs(h2[q])));
  }
  INFO("max relative device - host difference ", worst);
  CHECK(worst <= 3e-14);
}

TEST_CASE("mvgaussian cdf: derivative blocks against central differences, both forms, three regions") {
  const S s(3);
  for (int reg : {lower, upper, rect}) {
    check_law(D1(s), reg);
    check_law(D2(s), reg);
  }
}

TEST_CASE("mvgaussian cdf: values against log_mvn_rect, and y against the location") {
  const S s(3);
  const D1 g1(s);
  const D2 g2(s);
  const double th[6] = {0.1, 0.3, -0.2, 0.4, 0.2, -0.1};
  const double y[3] = {0.2, -0.4, 0.7}, mu[3] = {0.1, -0.1, 0.2};
  std::vector<double> w1(g1.state_size() + g1.scratch_size() + g1.cdf_scratch_size(2, 3));
  const auto st1 = g1.theta_state(th, w1.data());
  double* ws = w1.data() + g1.state_size();
  // Sigma of the covariance form
  std::vector<double> Sg(6), sw(64);
  g1.covariance(th, Sg.data(), sw.data());
  double lo[3], hi[3];
  for (int i = 0; i < 3; ++i) lo[i] = -I, hi[i] = y[i] - mu[i];
  std::vector<double> mw(laplace::math::mvn_rect_scratch(3));
  const double ref = laplace::math::log_mvn_rect(3, lo, hi, Sg.data(), mw.data());
  CHECK(std::fabs(g1.log_cdf(st1, y, mu, ws) - ref) <= 1e-15 * std::fabs(ref));
  CHECK(std::fabs(g1.cdf(st1, y, mu, ws) - std::exp(ref)) <= 1e-15);
  // the precision form at Q = Sigma^-1 of the same theta: the same law only when
  // M(theta) is inverted, so compare it against its own covariance
  std::vector<double> w2(g2.state_size() + g2.scratch_size() + g2.cdf_scratch_size(2, 3));
  const auto st2 = g2.theta_state(th, w2.data());
  std::vector<double> S2(6);
  g2.covariance(th, S2.data(), sw.data());
  const double ref2 = laplace::math::log_mvn_rect(3, lo, hi, S2.data(), mw.data());
  CHECK(std::fabs(g2.log_cdf(st2, y, mu, w2.data() + g2.state_size()) - ref2) <= 1e-13 * std::fabs(ref2));
  // the ccdf is the cdf of -Y at -y
  for (int i = 0; i < 3; ++i) lo[i] = y[i] - mu[i], hi[i] = I;
  const double refc = laplace::math::log_mvn_rect(3, lo, hi, Sg.data(), mw.data());
  CHECK(std::fabs(g1.log_ccdf(st1, y, mu, ws) - refc) <= 1e-15 * std::fabs(refc));
  // d_y^J d_mu^A = (-1)^J d_mu^(J + A): J = 1, A = 1 against A = 2
  const int idx[2] = {2, 4};
  std::vector<double> a(9), m2(6);
  g1.log_cdf_dy_dk<1, 1, 1>(st1, y, mu, idx, a.data(), ws);
  g1.log_cdf_dk<1, 2>(st1, y, mu, idx, m2.data(), ws);
  double e = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const int t[2] = {i < j ? i : j, i < j ? j : i};
      e = std::fmax(e, std::fabs(a[i * 3 + j] + m2[ld::detail_mv::block_position(3, 2, t)]));
    }
  CHECK(e == 0.0);
}

TEST_CASE("mvgaussian cdf: a singular precision has none") {
  const imp_t g(lp::weighted_sum_psdmat(std::vector<lp::weighted_sum_psdmat::host_matrix>{[] {
    lp::weighted_sum_psdmat::host_matrix m("S", 3, 3);
    // first differences: rank 2, null space the constant vector
    const double D[2][3] = {{-1, 1, 0}, {0, -1, 1}};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) m(i, j) = D[0][i] * D[0][j] + D[1][i] * D[1][j];
    return m;
  }()}));
  const double y[3] = {0, 0, 0}, mu[3] = {0, 0, 0};
  const int idx[1] = {0};
  double out[6];
  ld::mvgaussian_rank_deficient_state st{};
  CHECK(std::isnan(g.log_cdf(st, y, mu, nullptr)));
  CHECK(std::isnan(g.cdf(st, y, mu, nullptr)));
  CHECK(g.log_cdf_dk<1, 2>(st, y, mu, idx, out, nullptr));
  for (double v : out) CHECK(std::isnan(v));
}
