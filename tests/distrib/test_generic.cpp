// The generic routes of laplace::distrib (generic_*.hpp) on the four pilot
// distributions with their own members hidden (generic_hidden.hpp): quantile,
// discrete cdf family, expected information and its derivatives in both forms,
// moments, rng; each against the hidden closed form (the twin), on the host and
// inside a kernel.
#include <doctest/doctest.h>

#include "generic_hidden.hpp"

#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/logistic.hpp>
#include <laplace/distrib/mcp.hpp>
#include <laplace/distrib/scad.hpp>
#include <laplace/distrib/cauchy.hpp>
#include <laplace/distrib/poisson.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <vector>

namespace ld = laplace::distrib;
using laplace_test::hide;

namespace {

constexpr double eps = 0x1p-52;

// |a - b| in units of eps * max(|b|, floor)
double err(double a, double b, double floor = 0.0) {
  if (a == b) return 0.0;
  return std::fabs(a - b) / (eps * std::fmax(std::fabs(b), floor));
}

template <class D>
double quantile_rel_err(const D& d, double p, double q) {
  // the error divided by the change of the quantile caused by one ulp of the cdf there
  const double y = d.quantile(p);
  const double f = std::exp(d.logpdf(y));
  const double cond = std::fmax(std::fabs(y) * eps, std::fmin(p, 1.0 - p) * eps / f);
  return std::fabs(q - y) / cond;
}

template <class D>
double check_quantile(const D& d, double bound) {
  const auto h = hide(d);
  double worst = 0.0;
  for (double p : {1e-300, 1e-100, 1e-20, 1e-8, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 1.0 - 1e-8, 1.0 - 1e-15}) {
    const double y = d.quantile(p);
    if (!(std::fabs(y) > 1e-290) && y != 0.0) continue;  // the closed form underflows
    const double e1 = quantile_rel_err(d, p, ld::quantile_by_root(h, p));
    const double e2 = quantile_rel_err(d, p, ld::quantile_by_root_from_log(h, std::log(p)));
    const double e3 = quantile_rel_err(d, p, ld::quantile_by_root_from_log_upper(h, std::log1p(-p)));
    INFO("p = " << p << ", y = " << y);
    CHECK(e1 <= bound);
    CHECK(e2 <= bound);
    if (p >= 1e-15) CHECK(e3 <= bound);  // log1p(-p) carries no information on p below that
    worst = std::fmax(worst, std::fmax(e1, std::fmax(e2, p >= 1e-15 ? e3 : 0.0)));
  }
  // deep in the upper tail, from log(1 - p)
  for (double lq : {-50.0, -300.0}) {
    const double q = ld::quantile_by_root_from_log_upper(h, lq);
    CHECK(d.log_ccdf(q) == doctest::Approx(lq).epsilon(1e-12));
  }
  return worst;
}

// pair-major components of the K-th derivative of I, or I itself
template <int P, int K, class T>
std::vector<double> info_components(const T& t) {
  std::vector<double> r;
  if constexpr (K == 0) {
    for (int i = 0; i < ld::sym_tensor_size(P, 2); ++i) r.push_back(t[i]);
  } else {
    for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
      for (int i = 0; i < ld::sym_tensor_size(P, K); ++i) r.push_back(t.pair[p][i]);
  }
  return r;
}

// the largest error of the generic route against the closed form, relative to the
// largest closed-form component; NaN components must be exactly those expected
template <int K, ld::info_form F, class D>
double info_err(const D& d) {
  constexpr int P = D::n_params;
  const auto exact = info_components<P, K>(ld::distrib_expected_info<K>(d));
  const auto gen = info_components<P, K>(ld::expected_info_by_integral<K, F>(hide(d)));
  double scale = 0.0;
  for (double e : exact) scale = std::fmax(scale, std::fabs(e));
  double worst = 0.0;
  for (size_t i = 0; i < exact.size(); ++i) {
    if (std::isnan(gen[i])) return INFINITY;  // only smooth distributions come here
    worst = std::fmax(worst, std::fabs(gen[i] - exact[i]) / (eps * scale));
  }
  return worst;
}

template <ld::info_form F, class D>
std::vector<double> info_errs(const D& d) {
  return {info_err<0, F>(d), info_err<1, F>(d), info_err<2, F>(d), info_err<3, F>(d), info_err<4, F>(d)};
}

template <class D>
void check_info(const D& d, const char* name, double bound_h, double bound_s) {
  const auto eh = info_errs<ld::info_form::hessian>(d);
  const auto es = info_errs<ld::info_form::score>(d);
  for (int K = 0; K <= 4; ++K) {
    INFO(name << ", K = " << K << ": hessian " << eh[K] << ", score " << es[K] << " eps");
    CHECK(eh[K] <= bound_h);
    CHECK(es[K] <= bound_s);
  }
  std::printf("[generic info] %-24s hessian %8.1f %8.1f %8.1f %8.1f %8.1f | score %8.1f %8.1f %8.1f %8.1f %8.1f eps\n",
              name, eh[0], eh[1], eh[2], eh[3], eh[4], es[0], es[1], es[2], es[3], es[4]);
}

// one kernel of f over out, timed, against f on the host
template <class F>
void run_kernel(const Kokkos::View<double*>& out, const char* name, const F& f) {
  const int n = int(out.extent(0));
  const auto t0 = std::chrono::steady_clock::now();
  Kokkos::parallel_for("generic_kernel", n, KOKKOS_LAMBDA(const int i) { out(i) = f(i); });
  Kokkos::fence();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("[generic kernel] %-28s %9.4f s for %d threads\n", name, s, n);
  std::fflush(stdout);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    INFO(name << ", i = " << i);
    CHECK(h(i) == doctest::Approx(f(i)).epsilon(1e-11));
  }
}

}  // namespace

TEST_CASE("distrib generic: quantile by root against the closed forms") {
  double w = 0.0;
  w = std::fmax(w, check_quantile(ld::gaussian1_distrib{0.4, 1.3}, 8.0));
  w = std::fmax(w, check_quantile(ld::gaussian1_distrib{-20.0, 1e-3}, 8.0));
  w = std::fmax(w, check_quantile(ld::laplace1_distrib{0.4, 1.3}, 8.0));
  w = std::fmax(w, check_quantile(ld::gamma1_distrib{2.0, 0.3}, 64.0));
  w = std::fmax(w, check_quantile(ld::gamma1_distrib{1.5, 4.0}, 64.0));
  w = std::fmax(w, check_quantile(ld::gamma1_distrib{1.5, 0.02}, 64.0));
  std::printf("[generic quantile] continuous: worst %.2f (error / change caused by one ulp of y or of p)\n", w);
  // ends and invalid p
  const auto h = hide(ld::gamma1_distrib{2.0, 0.3});
  CHECK(ld::quantile_by_root(h, 0.0) == 0.0);
  CHECK(ld::quantile_by_root(h, 1.0) == ld::distrib_infinity);
  CHECK(std::isnan(ld::quantile_by_root(h, -0.1)));
  CHECK(std::isnan(ld::quantile_by_root(h, std::nan(""))));
  // the dispatcher takes the member when there is one
  const ld::gaussian1_distrib g{0.4, 1.3};
  CHECK(ld::distrib_quantile(g, 0.3) == g.quantile(0.3));
}

TEST_CASE("distrib generic: quantile by search (Poisson)") {
  for (double mu : {0.05, 3.5, 40.0, 1e4}) {
    const ld::poisson_distrib d{mu};
    const auto h = hide(d);
    for (double p : {1e-300, 1e-30, 1e-6, 0.01, 0.25, 0.5, 0.77, 0.99, 1.0 - 1e-9, 1.0 - 1e-15}) {
      const double k = ld::quantile_by_search(h, p);
      INFO("mu = " << mu << ", p = " << p << ", k = " << k);
      // the smallest k with F(k) >= p, read on the closed form (ties within 1e-13 skipped)
      const bool upper = p > 0.5;
      const double lt = upper ? std::log1p(-p) : std::log(p);
      auto ok = [&](double j) { return upper ? d.log_ccdf(j) <= lt : d.log_cdf(j) >= lt; };
      const double near = upper ? std::fabs(d.log_ccdf(k) - lt) : std::fabs(d.log_cdf(k) - lt);
      const double near1 = upper ? std::fabs(d.log_ccdf(k - 1) - lt) : std::fabs(d.log_cdf(k - 1) - lt);
      if (near > 1e-13 && near1 > 1e-13) {
        CHECK(ok(k));
        if (k > 0) CHECK(!ok(k - 1));
      }
      CHECK(ld::quantile_by_search_from_log(h, std::log(p)) == k);
      if (p > 1e-15) CHECK(ld::quantile_by_search_from_log_upper(h, std::log1p(-p)) == k);
    }
    CHECK(ld::quantile_by_search(h, 0.0) == 0.0);
    CHECK(ld::quantile_by_search(h, 1.0) == ld::distrib_infinity);
  }
}

TEST_CASE("distrib generic: Poisson cdf family by summing the mass") {
  for (double mu : {0.05, 3.5, 40.0, 1e3}) {
    double w = 0.0;
    const double bound = 16.0;
    const ld::poisson_distrib d{mu};
    const auto h = hide(d);
    const double s = std::sqrt(mu);
    for (double z : {-30.0, -8.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 8.0, 30.0, 100.0}) {
      const double k = std::floor(mu + z * s);
      if (k < 0) continue;
      INFO("mu = " << mu << ", k = " << k);
      const double e1 = err(ld::log_cdf_by_sum(h, k), d.log_cdf(k), 1.0);
      const double e2 = err(ld::log_ccdf_by_sum(h, k), d.log_ccdf(k), 1.0);
      const double e3 = err(ld::cdf_by_sum(h, k), d.cdf(k));
      const double e4 = err(ld::ccdf_by_sum(h, k), d.ccdf(k));
      // log F and log(1 - F) in ulps of max(|log F|, 1); F, 1 - F where above 1e-300
      CHECK(e1 <= bound);
      CHECK(e2 <= bound);
      if (d.cdf(k) > 1e-300) CHECK(e3 <= bound * std::fmax(1.0, std::fabs(d.log_cdf(k))));
      if (d.ccdf(k) > 1e-300) CHECK(e4 <= bound * std::fmax(1.0, std::fabs(d.log_ccdf(k))));
      w = std::fmax(w, std::fmax(e1, e2));
    }
    std::printf("[generic cdf] Poisson(%g) log F, log(1 - F): worst %.2f ulp of max(|log|, 1)\n", mu, w);
    CHECK(ld::cdf_by_sum(h, -1.0) == 0.0);
    CHECK(ld::ccdf_by_sum(h, -1.0) == 1.0);
    CHECK(ld::log_cdf_by_sum(h, 2.5) == ld::log_cdf_by_sum(h, 2.0));
  }
}

template <int K>
double sum_side(const ld::poisson_distrib& d, double k, bool upper) {
  return upper ? ld::log_ccdf_dk_by_sum<K>(d, k)[0] : ld::log_cdf_dk_by_sum<K>(d, k)[0];
}

TEST_CASE("distrib generic: Poisson log F derivatives by summing the mass, against mpmath") {
  // the references of tools/reference/distrib_log_cdf.py (the Poisson rows), each
  // error in ulps of the component itself, as in test_pilot.cpp. The route loses
  // about five digits at orders 5 and 6 near the median (the step from the sums
  // d_T S / S to the derivatives of log S cancels there, about 1e-10 relative,
  // without getting worse as the mean grows: docs/numerics.md).
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_log_cdf.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  // measured on 2026-11-15, worst ulp per order, the same on gcc and on CUDA: 476, 502, 856,
  // 7.58e3, 2.41e5, 5.34e5; bounds half again
  const double bound[7] = {0.0, 715.0, 755.0, 1.3e3, 1.14e4, 3.62e5, 8.0e5};
  double worst[7] = {};
  int rows = 0;
  while (std::fgets(line, sizeof line, f)) {
    int dist, upper, K, b;
    char ys[64], his[64], los[64], mus[64], phis[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &dist, &upper, ys, &K, &b, his, los, mus,
                    phis) != 9)
      continue;
    if (dist != 2) continue;
    const double y = std::strtod(ys, nullptr), hi = std::strtod(his, nullptr), lo = std::strtod(los, nullptr),
                 mu = std::strtod(mus, nullptr);
    const ld::poisson_distrib d{mu};
    double v = 0.0;
    switch (K) {
      case 1: v = sum_side<1>(d, y, upper); break;
      case 2: v = sum_side<2>(d, y, upper); break;
      case 3: v = sum_side<3>(d, y, upper); break;
      case 4: v = sum_side<4>(d, y, upper); break;
      case 5: v = sum_side<5>(d, y, upper); break;
      case 6: v = sum_side<6>(d, y, upper); break;
    }
    const double u = std::fabs((v - hi) - lo) / std::ldexp(1.0, std::ilogb(hi) - 52);
    INFO("mu = " << mu << ", k = " << y << ", K = " << K << ", upper = " << upper << ": " << u << " ulp");
    CHECK(u <= bound[K]);
    worst[K] = std::fmax(worst[K], u);
    ++rows;
  }
  std::fclose(f);
  CHECK(rows > 300);
  for (int K = 1; K <= 6; ++K)
    std::printf("[generic log F derivatives] Poisson, order %d: worst %.3g ulp of the component\n", K, worst[K]);
}

// The cdf family of a continuous distribution by quadrature of its density,
// against the closed forms of the Cauchy and of the logistic (the families whose
// log F and its six derivatives are written out), on both sides and at both ends
// of the range. The quantity compared is the error relative to max(|f|, 1): the
// pure derivatives in the scale vanish at y = mu, where a relative measure of
// the component itself has no meaning.
template <int K, class D>
double worst_log_cdf_dk_by_integral(const D& d) {
  double w = 0.0;
  const auto h = hide(d);
  for (double y : {-40.0, -9.0, -3.0, -1.0, -0.2, 0.0, 0.4, 2.0, 6.0, 25.0})
    for (int side = 0; side < 2; ++side) {
      const auto gen = side ? ld::distrib_log_ccdf_dk<K>(h, y) : ld::distrib_log_cdf_dk<K>(h, y);
      const auto own = side ? d.template log_ccdf_dk<K>(y) : d.template log_cdf_dk<K>(y);
      for (int i = 0; i < ld::sym_tensor<2, K>::size; ++i) {
        INFO("K = " << K << ", y = " << y << ", side = " << side << ", component " << i);
        const double e = std::fabs(gen[i] - own[i]) / std::fmax(std::fabs(own[i]), 1.0);
        CHECK(std::isfinite(gen[i]));
        w = std::fmax(w, e);
      }
    }
  return w;
}

TEST_CASE("distrib generic: continuous cdf family by quadrature of the density") {
  // the value, on both sides and into the tails where F underflows
  for (double sigma : {0.25, 1.0, 7.0}) {
    const ld::cauchy_distrib c{0.0, sigma};
    const ld::logistic_distrib g{0.0, sigma};
    const auto hc = hide(c);
    const auto hg = hide(g);
    for (double y : {-300.0, -40.0, -9.0, -1.0, 0.0, 0.4, 6.0, 25.0, 300.0}) {
      INFO("sigma = " << sigma << ", y = " << y);
      CHECK(err(ld::log_cdf_by_integral(hc, y), c.log_cdf(y), 1.0) <= 8.0);
      CHECK(err(ld::log_ccdf_by_integral(hc, y), c.log_ccdf(y), 1.0) <= 8.0);
      CHECK(err(ld::log_cdf_by_integral(hg, y), g.log_cdf(y), 1.0) <= 8.0);
      CHECK(err(ld::log_ccdf_by_integral(hg, y), g.log_ccdf(y), 1.0) <= 8.0);
    }
    CHECK(ld::cdf_by_integral(hc, -1e300) == 0.0);
    CHECK(ld::ccdf_by_integral(hc, 1e300) == 0.0);
  }
  // the derivatives, through the dispatcher (which takes this route for a
  // continuous distribution without a log_cdf_dk member); bounds at about three
  // times the error measured on 2026-11-15 on gcc and CUDA
  const ld::cauchy_distrib c{0.0, 0.25};
  const ld::logistic_distrib g{0.0, 0.25};
  const double w1 = worst_log_cdf_dk_by_integral<1>(c), w2 = worst_log_cdf_dk_by_integral<2>(c);
  const double w3 = worst_log_cdf_dk_by_integral<3>(c), w4 = worst_log_cdf_dk_by_integral<4>(c);
  const double w5 = worst_log_cdf_dk_by_integral<5>(c), w6 = worst_log_cdf_dk_by_integral<6>(c);
  const double wg = worst_log_cdf_dk_by_integral<6>(g);
  std::printf("[generic cdf] by quadrature, worst over max(|f|, 1): cauchy orders 1-6 %.1e %.1e %.1e %.1e %.1e %.1e, logistic order 6 %.1e\n",
              w1, w2, w3, w4, w5, w6, wg);
  CHECK(w1 <= 6e-15);
  CHECK(w2 <= 5e-14);
  CHECK(w3 <= 2e-13);
  CHECK(w4 <= 3e-12);
  CHECK(w5 <= 7e-11);
  CHECK(w6 <= 2e-09);
  CHECK(wg <= 5e-09);
}

TEST_CASE("distrib generic: log F derivatives by sum, ends of the support and dispatcher") {
  const ld::poisson_distrib d{3.5};
  const auto h = hide(d);
  CHECK(std::isnan(ld::log_cdf_dk_by_sum<2>(h, -1.0)[0]));
  CHECK(ld::log_ccdf_dk_by_sum<2>(h, -1.0)[0] == 0.0);
  CHECK(ld::log_cdf_dk_by_sum<3>(h, 2.5)[0] == ld::log_cdf_dk_by_sum<3>(h, 2.0)[0]);
  CHECK(ld::distrib_log_cdf_dk<4>(d, 2.0)[0] == d.log_cdf_d4(2.0)[0]);
  CHECK(ld::distrib_log_ccdf_dk<4>(d, 2.0)[0] == d.log_ccdf_d4(2.0)[0]);
  CHECK(ld::distrib_log_cdf_dk<4>(h, 2.0)[0] == ld::log_cdf_dk_by_sum<4>(h, 2.0)[0]);
}

TEST_CASE("distrib generic: expected information and its derivatives, both forms") {
  // smooth distributions: both forms against the closed forms
  check_info(ld::gaussian1_distrib{0.4, 1.3}, "gaussian1(0.4, 1.3)", 16.0, 16.0);
  for (double phi : {0.02, 0.3, 4.0}) {
    char name[64];
    std::snprintf(name, sizeof name, "gamma1(2, %g)", phi);
    check_info(ld::gamma1_distrib{2.0, phi}, name, 16.0, 16.0);
  }
  for (double mu : {0.05, 3.5, 40.0}) {
    char name[64];
    std::snprintf(name, sizeof name, "poisson(%g)", mu);
    // mu = 40: d3 and d4 I are 1/mu^4 and 1/mu^5 of terms of order 1 (cancellation)
    check_info(ld::poisson_distrib{mu}, name, mu < 10.0 ? 16.0 : 400.0, mu < 10.0 ? 24.0 : 200.0);
  }
}

TEST_CASE("distrib generic: expected information of the Laplace (finite regularity)") {
  const ld::laplace1_distrib d{0.4, 1.3};
  const auto h = hide(d);
  // I itself: the score form gives it, the hessian form only the (sigma, sigma) component
  const auto is = ld::expected_info_by_integral<0, ld::info_form::score>(h);
  const auto ih = ld::expected_info_by_integral<0, ld::info_form::hessian>(h);
  const auto e = d.expected_info();
  for (int i = 0; i < 3; ++i) CHECK(std::fabs(is[i] - e[i]) <= 64 * eps * e[0]);
  CHECK(std::isnan(ih[0]));  // -E[l_mumu] misses the point mass at the kink
  CHECK(std::fabs(ih[1]) <= 64 * eps * e[2]);
  CHECK(std::fabs(ih[2] - e[2]) <= 64 * eps * e[2]);
  // derivatives: pairs (mu, mu), (mu, sigma), (sigma, sigma); C sorted, position = number of sigma
  const auto d1 = ld::expected_info_by_integral<1, ld::info_form::score>(h);
  const auto x1 = d.expected_info_d1();
  CHECK(std::isnan(d1.pair[0][0]));  // d_mu I_mumu: mu in (a, b) and in C
  CHECK(std::isnan(d1.pair[1][0]));  // d_mu I_musigma
  CHECK(std::fabs(d1.pair[2][0] - x1.pair[2][0]) <= 1e-13);  // d_mu I_sigmasigma = 0
  for (int p = 0; p < 3; ++p) CHECK(std::fabs(d1.pair[p][1] - x1.pair[p][1]) <= 64 * eps * std::fabs(x1.pair[0][1]));
  const auto d2 = ld::expected_info_by_integral<2, ld::info_form::score>(h);
  const auto x2 = d.expected_info_d2();
  CHECK(std::isnan(d2.pair[2][0]));  // two mu in C
  CHECK(std::fabs(d2.pair[2][1] - x2.pair[2][1]) <= 1e-13);
  for (int p = 0; p < 3; ++p) CHECK(std::fabs(d2.pair[p][2] - x2.pair[p][2]) <= 256 * eps * std::fabs(x2.pair[0][2]));
  const auto d4 = ld::expected_info_by_integral<4, ld::info_form::score>(h);
  const auto x4 = d.expected_info_d4();
  for (int p = 0; p < 3; ++p) CHECK(std::fabs(d4.pair[p][4] - x4.pair[p][4]) <= 1024 * eps * std::fabs(x4.pair[0][4]));
  // the dispatcher: the member here, the generic route on the hidden one (score form)
  CHECK(ld::distrib_expected_info<0>(d)[0] == e[0]);
  CHECK(ld::distrib_expected_info<0>(h)[0] == is[0]);
}

TEST_CASE("distrib generic: moments by quadrature and sums") {
  auto rel = [](double a, double b, double s) { return std::fabs(a - b) / (eps * s); };
  double w = 0.0;
  {
    const auto h = hide(ld::gaussian1_distrib{0.4, 1.3});
    const double s2 = 1.3 * 1.3;
    w = std::fmax(w, rel(ld::moment_by_integral(h, 1), 0.4, 0.4));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 2, 0.4), s2, s2));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 3, 0.4), 0.0, s2 * 1.3));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 4, 0.4), 3 * s2 * s2, 3 * s2 * s2));
  }
  {
    const auto h = hide(ld::laplace1_distrib{0.4, 1.3});
    w = std::fmax(w, rel(ld::moment_by_integral(h, 1), 0.4, 0.4));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 2, 0.4), 2 * 1.69, 2 * 1.69));
  }
  std::printf("[generic moments] gaussian1, laplace: worst %.2f eps of the scale\n", w);
  CHECK(w <= 64.0);
  for (double mu : {0.05, 3.5, 40.0, 1e3}) {
    w = 0.0;
    const auto h = hide(ld::poisson_distrib{mu});
    w = std::fmax(w, rel(ld::moment_by_integral(h, 1), mu, mu));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 2, mu), mu, mu));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 3, mu), mu, std::pow(mu, 1.5)));
    std::printf("[generic moments] poisson(%g): worst %.2f eps of the scale\n", mu, w);
    CHECK(w <= 16.0);
  }
  w = 0.0;
  for (double phi : {0.02, 0.3, 4.0}) {
    const auto h = hide(ld::gamma1_distrib{2.0, phi});
    const double v = phi * 4.0;
    w = std::fmax(w, rel(ld::moment_by_integral(h, 1), 2.0, 2.0));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 2, 2.0), v, v));
    w = std::fmax(w, rel(ld::moment_by_integral(h, 3, 2.0), 2 * phi * phi * 8.0, std::pow(v, 1.5)));
  }
  std::printf("[generic moments] gamma1: worst %.2f eps of the scale\n", w);
  CHECK(w <= 64.0);
}

// The bodies that define device lambdas sit in named functions with external linkage,
// never in a TEST_CASE body: doctest's case function has internal linkage, and an extended
// device lambda takes the address of its enclosing function as a tag, which collides
// between translation units (note/brief_agenti.md, found on 2026-11-13).
void generic_rng_kernel_case() {
  const int n = 1 << 16;
  Kokkos::View<double*> out("draws", n);
  const ld::gamma1_distrib ga{2.0, 0.3};
  Kokkos::parallel_for(
      "generic_rng", n, KOKKOS_LAMBDA(const int i) {
        ld::counter_uniform_source src{12345u, std::uint64_t(i)};
        out(i) = ld::rng_by_inversion(hide(ga), src) + ld::rng_by_inversion(ld::poisson_distrib{3.5}, src);
      });
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double m = 0.0, v = 0.0;
  for (int i = 0; i < n; ++i) {
    ld::counter_uniform_source src{12345u, std::uint64_t(i)};
    const double r = ld::rng_by_inversion(hide(ga), src) + ld::rng_by_inversion(ld::poisson_distrib{3.5}, src);
    if (i < 256) CHECK(hv(i) == doctest::Approx(r).epsilon(1e-12));
    if (i < 256) {
      ld::counter_uniform_source first{12345u, std::uint64_t(i)};
      CHECK(ld::rng_by_inversion(hide(ga), 12345u, std::uint64_t(i)) == ld::rng_by_inversion(hide(ga), first));
    }
    m += hv(i);
  }
  m /= n;
  for (int i = 0; i < n; ++i) v += (hv(i) - m) * (hv(i) - m);
  v /= n - 1;
  // mean 2 + 3.5, variance 0.3 * 4 + 3.5; 5 standard errors
  CHECK(std::fabs(m - 5.5) <= 5.0 * std::sqrt(4.7 / n));
  CHECK(std::fabs(v - 4.7) <= 5.0 * 4.7 * std::sqrt(2.0 / n) * 1.5);
  // a Kokkos pool state works as the source
  Kokkos::Random_XorShift64_Pool<> pool(2026);
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "generic_rng_pool", n,
      KOKKOS_LAMBDA(const int, double& acc) {
        auto g = pool.get_state();
        acc += ld::rng_by_inversion(ld::laplace1_distrib{0.4, 1.3}, g);
        pool.free_state(g);
      },
      sum);
  CHECK(std::fabs(sum / n - 0.4) <= 5.0 * 1.3 * std::sqrt(2.0 / n));
}

TEST_CASE("distrib generic: rng by inversion, counter source reproducible on the device") { generic_rng_kernel_case(); }

void generic_routes_kernel_case() {
  const int n = 16;
  Kokkos::View<double*> out("out", n);
  const ld::gaussian1_distrib g{0.4, 1.3};
  const ld::laplace1_distrib l{0.4, 1.3};
  const ld::gamma1_distrib ga{2.0, 0.3};
  const ld::poisson_distrib p{3.5};
  run_kernel(out, "quantile_by_root gamma1", KOKKOS_LAMBDA(int i) { return ld::quantile_by_root(hide(ga), (i + 0.5) / n); });
  run_kernel(out, "quantile_by_root laplace", KOKKOS_LAMBDA(int i) { return ld::quantile_by_root(hide(l), (i + 0.5) / n); });
  run_kernel(out, "quantile_by_search poisson", KOKKOS_LAMBDA(int i) { return ld::quantile_by_search(hide(p), (i + 0.5) / n); });
  run_kernel(out, "log_cdf_by_sum poisson", KOKKOS_LAMBDA(int i) { return ld::log_cdf_by_sum(hide(p), double(i)); });
  run_kernel(out, "log_cdf_dk_by_sum poisson", KOKKOS_LAMBDA(int i) { return ld::log_cdf_dk_by_sum<6>(hide(p), double(i))[0]; });
  run_kernel(out, "log_ccdf_dk_by_sum poisson", KOKKOS_LAMBDA(int i) { return ld::log_ccdf_dk_by_sum<3>(hide(p), double(i))[0]; });
  run_kernel(out, "log_cdf_by_integral gaussian1", KOKKOS_LAMBDA(int i) {
    return ld::log_cdf_by_integral(hide(g), 0.4 + 0.2 * i);
  });
  run_kernel(out, "log_cdf_dk_by_integral gaussian1", KOKKOS_LAMBDA(int i) {
    return ld::log_cdf_dk_by_integral<2>(hide(g), 0.4 + 0.2 * i)[0];
  });
  run_kernel(out, "log_ccdf_dk_by_integral gaussian1", KOKKOS_LAMBDA(int i) {
    return ld::log_ccdf_dk_by_integral<3>(hide(g), 0.4 + 0.2 * i)[1];
  });
  run_kernel(out, "moment gamma1", KOKKOS_LAMBDA(int) { return ld::moment_by_integral(hide(ga), 2, 2.0); });
  run_kernel(out, "info d0 laplace score", KOKKOS_LAMBDA(int) {
    return ld::expected_info_by_integral<0, ld::info_form::score>(hide(l))[0];
  });
  run_kernel(out, "info d3 poisson score", KOKKOS_LAMBDA(int) {
    return ld::expected_info_by_integral<3, ld::info_form::score>(hide(p)).pair[0][0];
  });
  run_kernel(out, "info d2 gaussian1 hessian", KOKKOS_LAMBDA(int) {
    return ld::expected_info_by_integral<2, ld::info_form::hessian>(hide(g)).pair[2][2];
  });
  run_kernel(out, "info d4 gamma1 score", KOKKOS_LAMBDA(int) {
    return ld::expected_info_by_integral<4, ld::info_form::score>(hide(ga)).pair[2][4];
  });
}

TEST_CASE("distrib generic: every route inside a kernel") { generic_routes_kernel_case(); }

// A family whose starting value rests on a statistic that no other family uses:
// it declares start_value over the sample and computes that statistic itself,
// so that the summary is never built for it.
namespace {

struct circular_start_probe {
  double mu = 0.0;
  double mean_cos = 0.0;

  circular_start_probe start_value(const std::vector<double>& y) const {
    double c = 0.0;
    double s = 0.0;
    for (double v : y) {
      c += std::cos(v);
      s += std::sin(v);
    }
    const double n = static_cast<double>(y.size());
    return {std::atan2(s / n, c / n), c / n};
  }
};

}  // namespace

TEST_CASE("distrib generic: the start value from the sample, by either route") {
  const std::vector<double> y = {0.2, 0.5, 1.0, 1.5, 2.0, 2.6, 3.1};

  // a distrib with start_value(summary) only: the sample route summarizes once
  // and gives exactly what the summary route gives
  const ld::gaussian1_distrib g{0.0, 1.0};
  const ld::gaussian1_distrib a = ld::distrib_start_value(g, ld::summarize_sample(y));
  const ld::gaussian1_distrib b = ld::distrib_start_value(g, y);
  CHECK(b.mu == a.mu);
  CHECK(b.sigma == a.sigma);

  // the same from a View, which copies to the host once and delegates
  Kokkos::View<double*, Kokkos::HostSpace> yv("y", y.size());
  for (std::size_t i = 0; i < y.size(); ++i) yv(i) = y[i];
  const ld::gaussian1_distrib c =
      ld::distrib_start_value(g, Kokkos::View<const double*, Kokkos::HostSpace>(yv));
  CHECK(c.mu == a.mu);
  CHECK(c.sigma == a.sigma);

  // a distrib with start_value(sample): its own statistic, no summary between
  double want_cos = 0.0;
  double want_sin = 0.0;
  for (double v : y) {
    want_cos += std::cos(v);
    want_sin += std::sin(v);
  }
  const double n = static_cast<double>(y.size());
  const circular_start_probe p = ld::distrib_start_value(circular_start_probe{}, y);
  CHECK(p.mean_cos == want_cos / n);
  CHECK(p.mu == std::atan2(want_sin / n, want_cos / n));
}

namespace {

// The validity rule of the generic information when a kink moves with several
// parameters at once. On mcp and scad the knots y = mu +- gamma lambda (and
// mu +- a lambda) move with mu, lambda and the shape, and l is C^1 there: the
// counts are summed over the parameters of finite regularity, so a component with
// two derivatives in C among lambda and the shape and one more index of them in
// (a, b) is NaN, while each parameter counted alone would have admitted it.
template <class D>
D moving_make(const double* t) {
  if constexpr (D::n_params == 4) return D{t[0], t[1], t[2], t[3]};
  else return D{t[0], t[1], t[2], t[3], t[4]};
}

// the rule as it read with each parameter counted alone, for the laws where the two agree
template <class D, int K>
bool rule_one_parameter(int a, int b, const int* c, ld::info_form form) {
  for (int j = 0; j < D::n_params; ++j) {
    const int r = D::params[j].regularity;
    if (r >= ld::infinitely_smooth) continue;
    const int na = a == j, nb = b == j;
    int nc = 0;
    for (int i = 0; i < K; ++i) nc += c[i] == j;
    if (form == ld::info_form::hessian) {
      if (na + nb + nc > r + 1) return false;
    } else if (nc > 0 && (na > nb ? na : nb) + nc > r + 1) {
      return false;
    }
  }
  return true;
}

template <class D, int K>
int rule_unchanged_count() {
  constexpr int P = D::n_params;
  int n = 0;
  for (int a = 0; a < P; ++a)
    for (int b = a; b < P; ++b) {
      int c[K > 0 ? K : 1] = {};
      for (int q = 0; q < ld::sym_tensor_size(P, K); ++q) {
        for (const auto form : {ld::info_form::hessian, ld::info_form::score}) {
          CHECK(ld::detail_generic::info_component_valid<D, K>(a, b, c, form) ==
                rule_one_parameter<D, K>(a, b, c, form));
          ++n;
        }
        int i = K - 1;
        while (i >= 0 && c[i] == P - 1) --i;
        if (i >= 0) {
          ++c[i];
          for (int j = i + 1; j < K; ++j) c[j] = c[i];
        }
      }
    }
  return n;
}

// The order-2 generic score-form components on a law whose knots move with lambda and
// the shape (indices 1 and 2): NaN exactly where the summed counts exceed r + 1 = 2, and
// every finite one equal to central differences of the order-1 generic route, which
// is valid at every component that carries a derivative in the knot's parameters.
template <class D>
void moving_knot_case(const double* th, double& worst, int& n_nan, int& n_finite) {
  constexpr int P = D::n_params;
  const D d = moving_make<D>(th);
  const auto g2 = ld::expected_info_by_integral<2, ld::info_form::score>(d);
  for (int a = 0; a < P; ++a)
    for (int b = a; b < P; ++b) {
      const int ab[2] = {a, b};
      const int p = ld::sym_tensor_position<P, 2>(ab);
      for (int c0 = 0; c0 < P; ++c0)
        for (int c1 = c0; c1 < P; ++c1) {
          const int c[2] = {c0, c1};
          const double v = g2.pair[p][ld::sym_tensor_position<P, 2>(c)];
          // the groups: {mu} of regularity 0 (the kink at y = mu), {mu, lambda, shape} of 1
          const auto group_ok = [&](int last, int r) {
            const auto in = [&](int j) { return j <= last; };
            const int na = in(a), nb = in(b), nc = in(c0) + in(c1);
            return nc == 0 || (na > nb ? na : nb) + nc <= r + 1;
          };
          const bool valid = group_ok(0, 0) && group_ok(2, 1);
          INFO("pair (" << a << ", " << b << "), C = (" << c0 << ", " << c1 << ")");
          CHECK(std::isnan(v) == !valid);
          if (!valid) {
            ++n_nan;
            continue;
          }
          // d/dtheta_c0 of the order-1 component d_c1 I_ab, by a five-point difference
          const double h = 1e-3 * std::fmax(std::fabs(th[c0]), 1.0);
          double t[5], f[4];
          const double steps[4] = {-2.0, -1.0, 1.0, 2.0};
          for (int s = 0; s < 4; ++s) {
            for (int k = 0; k < P; ++k) t[k] = th[k];
            t[c0] += steps[s] * h;
            f[s] = ld::expected_info_by_integral<1, ld::info_form::score>(moving_make<D>(t)).pair[p][c1];
          }
          const double fd = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h);
          if (!std::isfinite(fd)) continue;  // the order-1 component is NaN itself
          const double e = std::fabs(v - fd) / std::fmax(std::fabs(fd), 1.0);
          worst = std::fmax(worst, e);
          ++n_finite;
          CHECK(e <= 1e-6);
        }
    }
}

}  // namespace

TEST_CASE("distrib generic: the information rule sums the counts over the parameters that move a kink") {
  // one parameter of finite regularity: the rule is the one-parameter rule exactly
  int n = rule_unchanged_count<ld::laplace1_distrib, 0>() + rule_unchanged_count<ld::laplace1_distrib, 1>() +
          rule_unchanged_count<ld::laplace1_distrib, 2>() + rule_unchanged_count<ld::laplace1_distrib, 3>() +
          rule_unchanged_count<ld::laplace1_distrib, 4>();
  CHECK(n > 0);

  // mcp and scad with the Student tail, at the point where the one-parameter rule had
  // given d^2 I_lambda_lambda / dgamma^2 at 2.32 times its value
  double worst = 0.0;
  int n_nan = 0, n_finite = 0;
  const double th_m[5] = {0.3, 1.3, 3.0, 0.9, 3.5}, th_s[5] = {0.3, 1.1, 3.7, 0.9, 3.5};
  moving_knot_case<ld::mcp_distrib<ld::student_tail>>(th_m, worst, n_nan, n_finite);
  moving_knot_case<ld::scad_distrib<ld::student_tail>>(th_s, worst, n_nan, n_finite);
  std::printf("generic info, moving knots: %d components NaN by the rule, %d finite, worst %.3g relative "
              "against differences of order one\n", n_nan, n_finite, worst);
  CHECK(n_finite > 0);
}
