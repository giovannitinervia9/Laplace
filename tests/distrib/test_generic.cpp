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
#include <laplace/distrib/laplace.hpp>
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
  w = std::fmax(w, check_quantile(ld::laplace_distrib{0.4, 1.3}, 8.0));
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
  const ld::laplace_distrib d{0.4, 1.3};
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
    const auto h = hide(ld::laplace_distrib{0.4, 1.3});
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

TEST_CASE("distrib generic: rng by inversion, counter source reproducible on the device") {
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
        acc += ld::rng_by_inversion(ld::laplace_distrib{0.4, 1.3}, g);
        pool.free_state(g);
      },
      sum);
  CHECK(std::fabs(sum / n - 0.4) <= 5.0 * 1.3 * std::sqrt(2.0 / n));
}

TEST_CASE("distrib generic: every route inside a kernel") {
  const int n = 16;
  Kokkos::View<double*> out("out", n);
  const ld::gaussian1_distrib g{0.4, 1.3};
  const ld::laplace_distrib l{0.4, 1.3};
  const ld::gamma1_distrib ga{2.0, 0.3};
  const ld::poisson_distrib p{3.5};
  run_kernel(out, "quantile_by_root gamma1", KOKKOS_LAMBDA(int i) { return ld::quantile_by_root(hide(ga), (i + 0.5) / n); });
  run_kernel(out, "quantile_by_root laplace", KOKKOS_LAMBDA(int i) { return ld::quantile_by_root(hide(l), (i + 0.5) / n); });
  run_kernel(out, "quantile_by_search poisson", KOKKOS_LAMBDA(int i) { return ld::quantile_by_search(hide(p), (i + 0.5) / n); });
  run_kernel(out, "log_cdf_by_sum poisson", KOKKOS_LAMBDA(int i) { return ld::log_cdf_by_sum(hide(p), double(i)); });
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
