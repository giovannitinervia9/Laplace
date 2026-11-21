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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace mvgaussian_cdf_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using S = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using D1 = ld::mvgaussian1_distrib<S>;
using D2 = ld::mvgaussian2_distrib<S>;
using imp_t = ld::mvgaussian2_distrib<lp::weighted_sum_psdmat>;
using deg_t = ld::mvgaussian1_distrib<lp::weighted_sum_psdmat>;

static_assert(ld::mv_has_cdf<D1> && ld::mv_has_cdf<D2> && ld::mv_has_cdf<imp_t> && ld::mv_has_cdf<deg_t>);

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

// log_rect_prob_dab_dk against mpmath (mvn_limit_derivs): one thread per row and split,
// every split Ja + Jb = 1..3 of the derivatives of log P in a and b, written in
// the reference's layout (orders 1..3, then the sorted tuples over (a, b))
using Dab = ld::mvgaussian1_distrib<S>;
constexpr int dab_nv = 8 + 36 + 120;

struct DabRows {
  Dab g2, g3, g4;
  Kokkos::View<double**, Kokkos::LayoutRight> in, out, w;  // in: k, a[4], b[4], theta[10]
  template <int Ja, int Jb>
  KOKKOS_INLINE_FUNCTION void split(const Dab& g, int k, const double* th, const double* a, const double* b, double* o,
                                    double* ws) const {
    const double mu[4] = {0, 0, 0, 0};
    double v[64];
    const auto st = g.theta_state(th, ws);
    g.template log_rect_prob_dab_dk<Ja, Jb, 0>(st, a, b, mu, nullptr, v, ws + g.state_size());
    const int z = 2 * k, n = Ja + Jb;
    int ua[4] = {0, 0, 0, 0}, ub[4] = {0, 0, 0, 0}, t[4], q = 0;
    do {
      for (int c = 0; c < Jb; ++c) ub[c] = 0;
      do {
        for (int c = 0; c < Ja; ++c) t[c] = ua[c];
        for (int c = 0; c < Jb; ++c) t[Ja + c] = k + ub[c];
        o[laplace::math::detail_mvnormal::tuples_upto(z, n - 1) - 1 + laplace::math::detail_mvnormal::tuple_rank(z, n, t)] =
            v[q++];
      } while (ld::detail_mv::next_block_tuple(k, Jb, ub));
    } while (ld::detail_mv::next_block_tuple(k, Ja, ua));
  }
  // one team of one thread per (row, split): on CUDA each is its own block, so
  // the rows' divergent routes do not run one after another inside a warp
  KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>::member_type& m) const {
    const int l = m.league_rank(), i = l / 9;
    const int k = int(in(i, 0));
    const double* a = &in(i, 1);
    const double* b = &in(i, 5);
    const double* th = &in(i, 9);
    const Dab& g = k == 2 ? g2 : k == 3 ? g3 : g4;
    double* o = &out(i, 0);
    double* ws = &w(l, 0);
    switch (l % 9) {
      case 0: split<1, 0>(g, k, th, a, b, o, ws); break;
      case 1: split<0, 1>(g, k, th, a, b, o, ws); break;
      case 2: split<2, 0>(g, k, th, a, b, o, ws); break;
      case 3: split<1, 1>(g, k, th, a, b, o, ws); break;
      case 4: split<0, 2>(g, k, th, a, b, o, ws); break;
      case 5: split<3, 0>(g, k, th, a, b, o, ws); break;
      case 6: split<2, 1>(g, k, th, a, b, o, ws); break;
      case 7: split<1, 2>(g, k, th, a, b, o, ws); break;
      default: split<0, 3>(g, k, th, a, b, o, ws);
    }
  }
};

}  // namespace mvgaussian_cdf_test

using namespace mvgaussian_cdf_test;

// the kernel in a function with external linkage (note/brief_agenti.md)
Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> mvgaussian_cdf_dab_rows(
    const std::vector<std::vector<double>>& rows) {
  const int n = int(rows.size());
  Kokkos::View<double**, Kokkos::LayoutRight> in("in", n, 19), out("out", n, dab_nv);
  auto hin = Kokkos::create_mirror_view(in);
  for (int i = 0; i < n; ++i) {
    const auto& r = rows[i];
    const int k = int(r[0]);
    hin(i, 0) = k;
    for (int c = 0; c < 4; ++c) hin(i, 1 + c) = r[1 + c], hin(i, 5 + c) = r[5 + c];
    // theta: the log-Cholesky coordinates of the one-factor correlation lam_i lam_j
    double L[4][4] = {};
    for (int p = 0; p < k; ++p)
      for (int q = 0; q <= p; ++q) {
        double s = p == q ? 1.0 : r[9 + p] * r[9 + q];
        for (int c = 0; c < q; ++c) s -= L[p][c] * L[q][c];
        L[p][q] = p == q ? std::sqrt(s) : s / L[q][q];
        hin(i, 9 + p * (p + 1) / 2 + q) = p == q ? std::log(L[p][q]) : L[p][q];
      }
  }
  Kokkos::deep_copy(in, hin);
  const Dab g2{S(2)}, g3{S(3)}, g4{S(4)};
  int sc = 0;
  for (int J = 1; J <= 3; ++J) sc = std::max(sc, g4.cdf_dab_scratch_size(J, 0, 0));
  Kokkos::View<double**, Kokkos::LayoutRight> w("w", 9 * n, g4.state_size() + g4.scratch_size() + sc);
  Kokkos::parallel_for("mvgaussian_cdf_dab", Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>(9 * n, 1),
                       DabRows{g2, g3, g4, in, out, w});
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

TEST_CASE("mvgaussian cdf: derivatives in the lower and upper limits against mpmath" * doctest::test_suite("slow")) {
  std::ifstream f(std::string(LAPLACE_REFERENCE_DIR) + "/mvn_limit_derivs.csv");
  REQUIRE(f.good());
  std::string line;
  std::getline(f, line);
  std::vector<std::vector<double>> rows;
  while (std::getline(f, line)) {
    std::vector<double> x;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) x.push_back(std::strtod(tok.c_str(), nullptr));
    if (!x.empty()) rows.push_back(x);
  }
  REQUIRE(rows.size() == 56);
  // On a device build the kernel runs 13 of the 56 rows (one GPU thread per row
  // in double precision took 3209 s for all of them); the host build checks every row.
  // Kept: far tail with both ends finite (0, 21, 42), with one infinite end
  // (1, 22, 43, 47), mixed (2, 23), and the bulk (14, 35, 52, 53); k = 2, 3, 4
  // each, every order and split in every row.
  if (!std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>) {
    std::vector<std::vector<double>> sub;
    for (int i : {0, 1, 2, 14, 21, 22, 23, 35, 42, 43, 47, 52, 53}) sub.push_back(rows[i]);
    rows.swap(sub);
  }
  const auto y = mvgaussian_cdf_dab_rows(rows);
  // worst relative error by order: rows with log P above -23; below it, the
  // entries in the limits of coordinates with one infinite end (the log route)
  // and those touching a coordinate with both ends finite (the recursion)
  namespace dm = laplace::math::detail_mvnormal;
  double worst[3][3] = {};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    const int k = int(r[0]), z = 2 * k;
    for (int n = 1; n <= 3; ++n) {
      const int o0 = dm::tuples_upto(z, n - 1) - 1;
      int t[3] = {0, 0, 0}, q = 0;
      do {
        bool two = false;
        for (int c = 0; c < n; ++c) two = two || (std::isfinite(r[1 + t[c] % k]) && std::isfinite(r[5 + t[c] % k]));
        const int cls = r[13] > -23 ? 0 : two ? 2 : 1;
        const double ref = r[15 + o0 + q], v = y(i, o0 + q);
        worst[cls][n - 1] = std::fmax(worst[cls][n - 1], std::fabs(v - ref) / std::fmax(1.0, std::fabs(ref)));
        ++q;
      } while (dm::next_tuple(z, n, t));
    }
  }
  INFO("log P > -23: orders 1-3 ", worst[0][0], " ", worst[0][1], " ", worst[0][2]);
  INFO("log P < -23, one infinite end: orders 1-3 ", worst[1][0], " ", worst[1][1], " ", worst[1][2]);
  INFO("log P < -23, both ends finite: orders 1-3 ", worst[2][0], " ", worst[2][1], " ", worst[2][2]);
  const double bound[3][3] = {{2e-14, 5.3e-12, 2.3e-10}, {9e-14, 3.5e-15, 2.8e-13}, {6.8e-13, 4.8e-10, 6.8e-8}};
  for (int c = 0; c < 3; ++c)
    for (int n = 0; n < 3; ++n) CHECK(worst[c][n] <= bound[c][n]);
}

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

TEST_CASE("mvgaussian cdf: theta blocks in the far tail, against central differences of the order below") {
  // bivariate, unstructured log-Cholesky, y deep in the lower tail, where the
  // blocks are assembled through the cumulants (heat equation for log F)
  using S2 = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
  const D1 g(S2(2));
  double th[3] = {0.1, 0.3, -0.2};
  double worst = 0.0;
  for (double t : {-6.0, -10.0, -15.0}) {
    double y[2] = {t, t + 0.5}, b[2] = {0, 0}, mu[2] = {0.0, 0.0};
    const int idx[3] = {1, 1, 2};
    worst = std::fmax(worst, diff_theta<D1, 1, 0>(g, lower, th, y, b, mu, idx + 2));
    worst = std::fmax(worst, diff_theta<D1, 2, 0>(g, lower, th, y, b, mu, idx + 1));
    worst = std::fmax(worst, diff_theta<D1, 3, 0>(g, lower, th, y, b, mu, idx));
    worst = std::fmax(worst, diff_theta<D1, 3, 2>(g, lower, th, y, b, mu, idx));
    worst = std::fmax(worst, diff_mu<D1, 2, 4>(g, lower, th, y, b, mu, idx));
  }
  INFO("max relative difference ", worst);
  CHECK(worst <= 1e-8);
}

TEST_CASE("mvgaussian cdf: four coordinates in the far tail, against central differences of the order below" *
          doctest::test_suite("slow")) {
  // the log route with the cumulants of the tensor rule over three coordinates;
  // the recursion lost 7.6e-4 (K = 3, A = 2), 3.7e-3 (A = 6) and 2.8e-2 (K = 2,
  // A = 4) at this point
  const D1 g(S(4));
  double th[10];
  for (int q = 0; q < 10; ++q) th[q] = 0.1 * ((q * 7) % 5) - 0.2;
  double y[4] = {-10.0, -9.7, -9.4, -9.1}, b[4] = {0, 0, 0, 0}, mu[4] = {0, 0, 0, 0};
  const int idx[3] = {1, 1, 2};
  const double e[] = {diff_theta<D1, 3, 2>(g, lower, th, y, b, mu, idx), diff_mu<D1, 2, 4>(g, lower, th, y, b, mu, idx),
                      diff_mu<D1, 0, 6>(g, lower, th, y, b, mu, idx)};
  for (double v : e) {
    INFO("relative difference ", v);
    CHECK(v <= 1e-8);
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

lp::weighted_sum_psdmat first_differences_3() {
  return lp::weighted_sum_psdmat(std::vector<lp::weighted_sum_psdmat::host_matrix>{[] {
    lp::weighted_sum_psdmat::host_matrix m("S", 3, 3);
    const double D[2][3] = {{-1, 1, 0}, {0, -1, 1}};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) m(i, j) = D[0][i] * D[0][j] + D[1][i] * D[1][j];
    return m;
  }()});
}

TEST_CASE("mvgaussian cdf: a singular covariance has none") {
  const deg_t g(first_differences_3());
  const double y[3] = {0, 0, 0}, mu[3] = {0, 0, 0};
  const int idx[1] = {0};
  double out[6];
  ld::mvgaussian_rank_deficient_state st{};
  CHECK(std::isnan(g.log_cdf(st, y, mu, nullptr)));
  CHECK(std::isnan(g.log_rect_prob(st, y, y, mu, nullptr)));
  CHECK(g.log_ccdf_dk<1, 2>(st, y, mu, idx, out, nullptr));
  for (double v : out) CHECK(std::isnan(v));
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
