// Speed of the generic routes of laplace::distrib (generic_*.hpp) against the
// pilot distributions' own members: quantile, discrete cdf, expected information
// and its derivatives in the hessian and the score form, moments, the uniform
// sources for rng. ns per call on one host thread (Serial) and on the default
// execution space. The accuracy is measured by tests/distrib/test_generic.cpp.
//
//   distrib_generic
#include "measure.hpp"

#include "../tests/distrib/generic_hidden.hpp"

#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/poisson.hpp>

#include <Kokkos_Random.hpp>

#include <cstdio>
#include <random>

namespace ld = laplace::distrib;
using laplace_test::hide;

namespace {

// a distribution at parameter t (the second parameter varied by the argument)
struct make_gaussian {
  KOKKOS_INLINE_FUNCTION ld::gaussian1_distrib operator()(double t) const { return {0.4, t}; }
};
struct make_laplace {
  KOKKOS_INLINE_FUNCTION ld::laplace1_distrib operator()(double t) const { return {0.4, t}; }
};
struct make_gamma {
  KOKKOS_INLINE_FUNCTION ld::gamma1_distrib operator()(double t) const { return {2.0, t}; }
};
struct make_poisson {
  KOKKOS_INLINE_FUNCTION ld::poisson_distrib operator()(double t) const { return {t}; }
};

// quantile at p of make(t): member (route 0) or generic (1)
template <class M>
struct quantile_f {
  M make;
  int route;
  KOKKOS_INLINE_FUNCTION double operator()(double p, double t) const {
    const auto d = make(t);
    return route == 0 ? ld::distrib_quantile(d, p) : ld::distrib_quantile(hide(d), p);
  }
};

struct poisson_cdf_f {
  int route;
  KOKKOS_INLINE_FUNCTION double operator()(double k, double mu) const {
    const ld::poisson_distrib d{mu};
    return route == 0 ? d.log_cdf(k) : ld::log_cdf_by_sum(hide(d), k);
  }
};

template <int P, int K, class T>
KOKKOS_INLINE_FUNCTION double sum_of(const T& t) {
  double s = 0.0;
  if constexpr (K == 0) {
    for (int i = 0; i < ld::sym_tensor_size(P, 2); ++i) s += t[i];
  } else {
    for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
      for (int i = 0; i < ld::sym_tensor_size(P, K); ++i) s += t.pair[p][i];
  }
  return s;
}

// d^K I of make(t): member (route 0), hessian form (1), score form (2)
template <class M, int K>
struct info_f {
  M make;
  int route;
  KOKKOS_INLINE_FUNCTION double operator()(double t) const {
    const auto d = make(t);
    constexpr int P = decltype(d)::n_params;
    if (route == 0) return sum_of<P, K>(ld::distrib_expected_info<K>(d));
    if (route == 1) return sum_of<P, K>(ld::expected_info_by_integral<K, ld::info_form::hessian>(hide(d)));
    return sum_of<P, K>(ld::expected_info_by_integral<K, ld::info_form::score>(hide(d)));
  }
};

struct moment_f {
  KOKKOS_INLINE_FUNCTION double operator()(double t) const { return ld::moment_by_integral(hide(ld::gamma1_distrib{2.0, t}), 2, 2.0); }
};

template <class F>
void time_line(const char* what, const char* route, F f, const bench::TimeArgs& t) {
  std::printf("%-34s %-8s | ns/call Serial %11.1f, %s %10.2f\n", what, route, bench::time_ns<bench::HostOne>(t.host, f, 3),
              bench::Exec::name(), bench::time_ns<bench::Exec>(t.dev, f, 5));
}

// a pool belongs to the default execution space: timed there only
template <class F>
void time_line_exec(const char* what, const char* route, F f, const bench::TimeArgs& t) {
  std::printf("%-34s %-8s | ns/call Serial           -, %s %10.2f\n", what, route, bench::Exec::name(),
              bench::time_ns<bench::Exec>(t.dev, f, 5));
}

template <class M>
void quantile_lines(const char* name, M make, double t0, bool generic_only = false) {
  std::mt19937_64 g(20261112);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const auto t = bench::make_time_args(1 << 12, [&] { return bench::Args{u(g), t0 * (0.9 + 0.2 * u(g)), 0.0}; });
  char what[64];
  std::snprintf(what, sizeof what, "quantile %s", name);
  if (!generic_only) time_line(what, "member", quantile_f<M>{make, 0}, t);
  time_line(what, "generic", quantile_f<M>{make, 1}, t);
}

template <class M, int K>
void info_lines(const char* name, M make, double t0) {
  std::mt19937_64 g(20261112);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const auto t = bench::make_time_args(1 << 8, [&] { return t0 * (0.9 + 0.2 * u(g)); });
  char what[64];
  std::snprintf(what, sizeof what, "info d%d %s", K, name);
  time_line(what, "member", info_f<M, K>{make, 0}, t);
  time_line(what, "hessian", info_f<M, K>{make, 1}, t);
  time_line(what, "score", info_f<M, K>{make, 2}, t);
}

template <class M>
void info_all(const char* name, M make, double t0) {
  info_lines<M, 0>(name, make, t0);
  info_lines<M, 1>(name, make, t0);
  info_lines<M, 2>(name, make, t0);
  info_lines<M, 3>(name, make, t0);
  info_lines<M, 4>(name, make, t0);
}

// evaluations of l per component of I (one quadrature or sum)
template <class D>
void evals_line(const char* name, const D& d) {
  const auto r = ld::expectation_by_integral(hide(d), [](double) { return 1.0; });
  std::printf("evaluations per component %-14s %d (converged %d)\n", name, r.evals, int(r.converged));
}

// the uniform sources: 1 draw per call
struct counter_f {
  KOKKOS_INLINE_FUNCTION double operator()(double i) const {
    ld::counter_uniform_source s{2026u, std::uint64_t(i)};
    return ld::open_unit_uniform(s);
  }
};
template <class Pool>
struct pool_f {
  Pool pool;
  KOKKOS_INLINE_FUNCTION double operator()(double) const {
    auto g = pool.get_state();
    const double u = ld::open_unit_uniform(g);
    pool.free_state(g);
    return u;
  }
};
struct rng_gamma_f {
  KOKKOS_INLINE_FUNCTION double operator()(double i) const {
    ld::counter_uniform_source s{2026u, std::uint64_t(i)};
    return ld::rng_by_inversion(ld::gamma1_distrib{2.0, 0.3}, s);
  }
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    quantile_lines("gaussian1(0.4, 1.3)", make_gaussian{}, 1.3);
    quantile_lines("laplace(0.4, 1.3)", make_laplace{}, 1.3);
    quantile_lines("gamma1(2, 0.3)", make_gamma{}, 0.3);
    quantile_lines("gamma1(2, 4)", make_gamma{}, 4.0);
    quantile_lines("poisson(3.5), member cdf", make_poisson{}, 3.5, true);
    {
      std::mt19937_64 g(20261112);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      for (double mu : {3.5, 40.0, 1e3}) {
        const auto t = bench::make_time_args(1 << 12, [&] {
          return bench::Args{std::floor(mu + std::sqrt(mu) * (6.0 * u(g) - 3.0)), mu, 0.0};
        });
        char what[64];
        std::snprintf(what, sizeof what, "log cdf poisson(%g), |z| < 3", mu);
        time_line(what, "member", poisson_cdf_f{0}, t);
        time_line(what, "sum", poisson_cdf_f{1}, t);
      }
    }
    evals_line("gaussian1", ld::gaussian1_distrib{0.4, 1.3});
    evals_line("laplace1", ld::laplace1_distrib{0.4, 1.3});
    evals_line("gamma1(2, 0.3)", ld::gamma1_distrib{2.0, 0.3});
    evals_line("gamma1(2, 4)", ld::gamma1_distrib{2.0, 4.0});
    evals_line("poisson(3.5)", ld::poisson_distrib{3.5});
    evals_line("poisson(40)", ld::poisson_distrib{40.0});
    info_all("gaussian1(0.4, 1.3)", make_gaussian{}, 1.3);
    info_all("gamma1(2, 0.3)", make_gamma{}, 0.3);
    info_all("poisson(3.5)", make_poisson{}, 3.5);
    info_lines<make_laplace, 0>("laplace(0.4, 1.3)", make_laplace{}, 1.3);
    {
      std::mt19937_64 g(20261112);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      const auto t = bench::make_time_args(1 << 8, [&] { return 0.3 * (0.9 + 0.2 * u(g)); });
      time_line("variance gamma1(2, 0.3)", "integral", moment_f{}, t);
    }
    {
      const auto t = bench::make_time_args(1 << 20, [i = 0.0]() mutable { return i++; });
      time_line("uniform draw", "counter", counter_f{}, t);
      time_line_exec("uniform draw", "xorshift64", pool_f<Kokkos::Random_XorShift64_Pool<>>{Kokkos::Random_XorShift64_Pool<>(2026)}, t);
      time_line_exec("uniform draw", "xs1024", pool_f<Kokkos::Random_XorShift1024_Pool<>>{Kokkos::Random_XorShift1024_Pool<>(2026)}, t);
      const auto t2 = bench::make_time_args(1 << 14, [i = 0.0]() mutable { return i++; });
      time_line("rng gamma1(2, 0.3) by inversion", "counter", rng_gamma_f{}, t2);
    }
  }
  Kokkos::finalize();
  return 0;
}
