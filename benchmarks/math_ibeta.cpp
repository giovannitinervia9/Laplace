// Accuracy and speed of the incomplete beta function: Laplace's own version
// with DiDonato and Morris' expansions (include/laplace/math/detail/ibeta_expansions.hpp,
// "dm"), the first own version (continued fraction only, frozen below as
// "cf"), Boost.Math in double ("boost") and, on the gcc host, Boost.Math in
// long double ("boost-ld").
//
// Usage: math_ibeta <reference dir> [mode [basym_min basym_lambda bup_max_a bgrat_min_a bpser_max_bx]]
//   mode all (default): accuracy and timing; accuracy: accuracy only; worst:
//   accuracy and, on the host, the worst arguments of "dm" per method;
//   methods, methods-mixed, methods-large: timing of dm and cf per method of
//   dm, on the common, mixed or large distribution of the timing below;
//   seq, seqr, seqc, seqrc <method>...: kernels per method (see main). The
//   thresholds of the choice (detail_ibx::thresholds) can be changed for
//   measurement (not in the seq modes, whose arguments are methods).
// References: ibeta, ibetac (incomplete.py), ibeta_hard, ibetac_hard (ibeta_hard.py).
#include "measure.hpp"

#include <laplace/math/detail/ibeta_expansions.hpp>
#include <laplace/math/incomplete_beta.hpp>

#include <boost/math/special_functions/beta.hpp>

#include <map>
#include <random>

namespace {

using laplace::math::detail::boost_policy;

struct result4 {
  double i, ic, log_i, log_ic;
};

// The first own version (2026-11-09, continued fraction of Numerical Recipes
// with the switch at (a + 1)/(a + b + 2)), frozen here for comparison.
namespace cf {
namespace lm = laplace::math;
KOKKOS_INLINE_FUNCTION double log1pmx_from(double t, double log_ratio) {
  if (Kokkos::fabs(t) <= 0.5) return boost::math::log1pmx(t, boost_policy());
  return log_ratio - t;
}
KOKKOS_INLINE_FUNCTION double log_prefactor(double a, double b, double x, double y) {
  if (a >= 10.0 && b >= 10.0) {
    using lm::own::detail_igamma::half_log_two_pi;
    using lm::own::detail_igamma::log_gamma_star;
    const double s = a + b, p0 = a / s, q0 = b / s;
    const double u = (x - p0) / p0, v = (p0 - x) / q0;
    const double Lu = log1pmx_from(u, lm::log(x / p0));
    const double Lv = log1pmx_from(v, lm::log(y / q0));
    return a * Lu + b * Lv + 0.5 * lm::log(a * b / s) - half_log_two_pi + log_gamma_star(s) - log_gamma_star(a) -
           log_gamma_star(b);
  }
  const double B = boost::math::beta(a, b, boost_policy());
  const double log_b = (B > 0.0 && B < lm::detail::pos_inf()) ? lm::log(B) : lm::lgamma(a) + lm::lgamma(b) - lm::lgamma(a + b);
  const double log_y = x < 0.5 ? Kokkos::log1p(-x) : lm::log(y);
  return a * lm::log(x) + b * log_y - log_b;
}
KOKKOS_INLINE_FUNCTION double continued_fraction(double a, double b, double x) {
  constexpr double eps = 1.1102230246251565e-16;
  constexpr double tiny = 1e-300;
  const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (Kokkos::fabs(d) < tiny) d = tiny;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m < 1000000; ++m) {
    const double m2 = 2.0 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (Kokkos::fabs(d) < tiny) d = tiny;
    c = 1.0 + aa / c;
    if (Kokkos::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (Kokkos::fabs(d) < tiny) d = tiny;
    c = 1.0 + aa / c;
    if (Kokkos::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (Kokkos::fabs(del - 1.0) <= 4.0 * eps) break;
  }
  return h;
}
KOKKOS_INLINE_FUNCTION result4 beta_inc(double a, double b, double x) {
  const double nan = lm::detail::quiet_nan();
  const double inf = lm::detail::pos_inf();
  if (!(a > 0.0) || !(b > 0.0) || !(x >= 0.0) || !(x <= 1.0)) return {nan, nan, nan, nan};
  if (x == 0.0) return {0.0, 1.0, -inf, 0.0};
  if (x == 1.0) return {1.0, 0.0, 0.0, -inf};
  const double y = 1.0 - x;
  const bool swap = x > (a + 1.0) / (a + b + 2.0);
  const double A = swap ? b : a, B = swap ? a : b, X = swap ? y : x, Y = swap ? x : y;
  const double log_small = log_prefactor(A, B, X, Y) - lm::log(A) + lm::log(continued_fraction(A, B, X));
  const double small = lm::exp(log_small);
  const double big = 1.0 - small;
  const double log_big = Kokkos::log1p(-small);
  if (!swap) return {small, big, log_small, log_big};
  return {big, small, log_big, log_small};
}
}  // namespace cf

// The new version, assembled from detail_ibx::direct_side
KOKKOS_INLINE_FUNCTION result4 dm_beta_inc(double a, double b, double x,
                                           const laplace::math::own::detail_ibx::thresholds& th = {}) {
  namespace lm = laplace::math;
  const double nan = lm::detail::quiet_nan();
  const double inf = lm::detail::pos_inf();
  if (!(a > 0.0) || !(b > 0.0) || !(x >= 0.0) || !(x <= 1.0)) return {nan, nan, nan, nan};
  if (x == 0.0) return {0.0, 1.0, -inf, 0.0};
  if (x == 1.0) return {1.0, 0.0, 0.0, -inf};
  const auto r = lm::own::detail_ibx::direct_side(a, b, x, th);
  const double w = lm::exp(r.log_w);
  const double other = 1.0 - w, log_other = Kokkos::log1p(-w);
  if (!r.complement) return {w, other, r.log_w, log_other};
  return {other, w, log_other, r.log_w};
}

// component k of an implementation: 0 i, 1 ic, 2 log_i, 3 log_ic
template <int K>
KOKKOS_INLINE_FUNCTION double pick(const result4& r) {
  if constexpr (K == 0) return r.i;
  else if constexpr (K == 1) return r.ic;
  else if constexpr (K == 2) return r.log_i;
  else return r.log_ic;
}
// thresholds of the choice, settable from the command line for measurement
laplace::math::own::detail_ibx::thresholds g_th;
template <int K>
struct Dm {
  laplace::math::own::detail_ibx::thresholds th = g_th;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return pick<K>(dm_beta_inc(a, b, x, th));
  }
};
template <int K>
struct Cf {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const { return pick<K>(cf::beta_inc(a, b, x)); }
};
// what laplace::math::ibeta / ibetac call today
template <int K>
struct Current {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    const auto r = laplace::math::own::beta_inc(a, b, x);
    return pick<K>(result4{r.i, r.ic, r.log_i, r.log_ic});
  }
};
template <int K>
struct Boost {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return K == 0 ? boost::math::ibeta(a, b, x, boost_policy()) : boost::math::ibetac(a, b, x, boost_policy());
  }
};
#ifndef KOKKOS_ENABLE_CUDA
namespace bp = boost::math::policies;
using promote_policy = bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>,
                                  bp::overflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>>;
template <int K>
struct BoostLd {
  double operator()(double a, double b, double x) const {
    return K == 0 ? boost::math::ibeta(a, b, x, promote_policy()) : boost::math::ibetac(a, b, x, promote_policy());
  }
};
#endif

// error of a logarithm output against log(hi + lo), in ulps of max(|log f|, 1)
struct LogAccuracy {
  double max_err = 0;
  bench::Args worst{0, 0, 0};
  std::size_t n_bad = 0;
};
LogAccuracy log_accuracy(const bench::Ref& r, const std::vector<double>& y) {
  LogAccuracy la;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!(r.hi[i] > 1e-300)) continue;  // the log of a subnormal reference is not accurate
    const double ref = std::log(r.hi[i]) + std::log1p(r.lo[i] / r.hi[i]);
    if (!std::isfinite(y[i])) {
      ++la.n_bad;
      continue;
    }
    const double e = std::fabs(y[i] - ref) / bench::ulp_of(std::max(std::fabs(ref), 1.0));
    if (e > la.max_err) {
      la.max_err = e;
      la.worst = r.x[i];
    }
  }
  return la;
}

template <class F>
void accuracy_line(const char* fn, const char* impl, const bench::Ref& r, F f, bool device = true) {
  const auto y = device ? bench::eval_kernel(r.x, f) : bench::eval_host(r.x, f);
  const auto a = bench::accuracy(r, y);
  std::printf("%-12s %-9s %-6s | max %9.4g ulp, %8.4g ulp1, %8.4g ulpL, mean %.3f | worst (%s) | non-finite %zu\n", fn,
              impl, device ? "kernel" : "host", a.max_ulp, a.max_ulp1, a.max_ulpL, a.mean_ulp,
              bench::show(a.worst, 3).c_str(), a.n_bad);
}

template <class F>
void log_line(const char* fn, const char* impl, const bench::Ref& r, F f) {
  const auto la = log_accuracy(r, bench::eval_kernel(r.x, f));
  std::printf("%-12s %-9s log    | max %9.4g ulp of max(|log f|, 1) | worst (%s) | non-finite %zu\n", fn, impl,
              la.max_err, bench::show(la.worst, 3).c_str(), la.n_bad);
}

// the worst arguments of dm, on the host, grouped by method
void worst_by_method(const char* fn, const bench::Ref& r, bool complement) {
  struct W {
    double ulp1 = 0, ulpL = 0;
    bench::Args a1{}, aL{};
    int n = 0;
  };
  std::map<int, W> by;
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    const auto [a, b, x] = r.x[i];
    if (!(x > 0.0 && x < 1.0)) continue;
    const auto s = laplace::math::own::detail_ibx::direct_side(a, b, x, g_th);
    const auto v = dm_beta_inc(a, b, x, g_th);
    const double y = complement ? v.ic : v.i;
    const double e = std::fabs(bench::error_in(y, r.hi[i], r.lo[i], r.hi[i]));
    const double e1 = std::fabs(bench::error_in(y, r.hi[i], r.lo[i], std::max(std::fabs(r.hi[i]), 1.0)));
    const double eL = r.hi[i] > 0 ? e / std::max(1.0, std::fabs(std::log(r.hi[i]))) : 0.0;
    W& w = by[s.method];
    ++w.n;
    if (e1 > w.ulp1) {
      w.ulp1 = e1;
      w.a1 = r.x[i];
    }
    if (eL > w.ulpL) {
      w.ulpL = eL;
      w.aL = r.x[i];
    }
  }
  for (const auto& [m, w] : by)
    std::printf("  %-12s method %d: n %5d | max ulp1 %8.3g at (%.17g, %.17g, %.17g) | max ulpL %8.3g at (%.17g, %.17g, %.17g)\n",
                fn, m, w.n, w.ulp1, w.a1[0], w.a1[1], w.a1[2], w.ulpL, w.aL[0], w.aL[1], w.aL[2]);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_ibeta <reference dir> [mode [thresholds]]\n");
    return 2;
  }
  const std::string refdir = argv[1];
  const bool worst = argc > 2 && std::string(argv[2]) == "worst";
  // optional: thresholds basym_min, basym_lambda, bup_max_a, bgrat_min_a (not for the debugging modes)
  const std::string mode0 = argc > 2 ? argv[2] : "all";
  const bool debug_mode = mode0.rfind("seq", 0) == 0;  // its arguments are methods
  if (!debug_mode && argc > 3) g_th.basym_min = std::atof(argv[3]);
  if (!debug_mode && argc > 4) g_th.basym_lambda = std::atof(argv[4]);
  if (!debug_mode && argc > 5) g_th.bup_max_a = std::atof(argv[5]);
  if (!debug_mode && argc > 6) g_th.bgrat_min_a = std::atof(argv[6]);
  if (!debug_mode && argc > 7) g_th.bpser_max_bx = std::atof(argv[7]);
  std::printf("thresholds: basym_min %g, basym_lambda %g, bup_max_a %g, bgrat_min_a %g, bpser_max_bx %g\n",
              g_th.basym_min, g_th.basym_lambda, g_th.bup_max_a, g_th.bgrat_min_a, g_th.bpser_max_bx);
  bench::print_spaces();

  const char* names[] = {"ibeta", "ibetac", "ibeta_hard", "ibetac_hard"};
  const bool skip_accuracy = argc > 2 && (std::string(argv[2]).rfind("methods", 0) == 0 || std::string(argv[2]).rfind("seq", 0) == 0);
  for (const char* name : names) {
    if (skip_accuracy) break;
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const bool comp = std::string(name).find("ibetac") == 0;
    if (!comp) {
      accuracy_line(name, "dm", r, Dm<0>{});
      accuracy_line(name, "dm", r, Dm<0>{}, false);
      accuracy_line(name, "current", r, Current<0>{});
      accuracy_line(name, "cf", r, Cf<0>{});
      accuracy_line(name, "boost", r, Boost<0>{});
#ifndef KOKKOS_ENABLE_CUDA
      accuracy_line(name, "boost-ld", r, BoostLd<0>{}, false);
#endif
      log_line(name, "dm", r, Dm<2>{});
      log_line(name, "cf", r, Cf<2>{});
    } else {
      accuracy_line(name, "dm", r, Dm<1>{});
      accuracy_line(name, "dm", r, Dm<1>{}, false);
      accuracy_line(name, "current", r, Current<1>{});
      accuracy_line(name, "cf", r, Cf<1>{});
      accuracy_line(name, "boost", r, Boost<1>{});
#ifndef KOKKOS_ENABLE_CUDA
      accuracy_line(name, "boost-ld", r, BoostLd<1>{}, false);
#endif
      log_line(name, "dm", r, Dm<3>{});
      log_line(name, "cf", r, Cf<3>{});
    }
    if (worst) worst_by_method(name, r, comp);
  }
  if (worst || (argc > 2 && std::string(argv[2]) == "accuracy")) return 0;  // accuracy only

  // Speed, ns per call: three distributions of the arguments
  //   common: a, b log-uniform in [1/4, 256], x uniform (as math_incomplete);
  //   mixed:  one shape log-uniform in [1/8, 32], the other in [64, 8192], x within 3 sd of the mean;
  //   large:  both log-uniform in [64, 8192], x within 3 sd of the mean.
  const std::size_t n = std::size_t(1) << 18;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };
  auto near_mean = [&](double a, double b) {
    const double s = a + b, m = a / s, sd = std::sqrt(a * b / (s * s * (s + 1)));
    double x = m + (6.0 * u01(gen) - 3.0) * sd;
    return std::min(std::max(x, 1e-300), 1.0 - 1e-16);
  };
  bool flip = false;
  const auto t_common = bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), u01(gen)}; });
  const auto t_mixed = bench::make_time_args(n, [&] {
    double a = expo(-3, 5), b = expo(6, 13);
    if ((flip = !flip)) std::swap(a, b);
    return bench::Args{a, b, near_mean(a, b)};
  });
  const auto t_large = bench::make_time_args(n, [&] {
    const double a = expo(6, 13), b = expo(6, 13);
    return bench::Args{a, b, near_mean(a, b)};
  });
  auto timing = [&](const char* impl, auto f, bool device = true) {
    std::printf("time %-9s | ns/call Serial: common %8.2f, mixed %8.2f, large %8.2f", impl,
                bench::time_ns<bench::HostOne>(t_common.host, f, 5), bench::time_ns<bench::HostOne>(t_mixed.host, f, 5),
                bench::time_ns<bench::HostOne>(t_large.host, f, 5));
    if (device)
      std::printf(" | %s: common %7.3f, mixed %7.3f, large %7.3f", bench::Exec::name(),
                  bench::time_ns<bench::Exec>(t_common.dev, f, 7), bench::time_ns<bench::Exec>(t_mixed.dev, f, 7),
                  bench::time_ns<bench::Exec>(t_large.dev, f, 7));
    std::printf("\n");
  };
  if (argc > 2 && std::string(argv[2]).rfind("seq", 0) == 0) {
    // Evaluate in kernels, one after the other, the arguments of the common
    // distribution sent to each method listed in argv 3...; seqr also times dm
    // with a parallel_reduce, seqc cf, seqrc both. "seqr 4 7" reproduced the
    // cudaErrorIllegalInstruction of an inlined direct_side inside a
    // parallel_reduce (see detail/ibeta_expansions.hpp).
    for (int k = 3; k < argc; ++k) {
      const int m = std::atoi(argv[k]);
      std::vector<bench::Args> v;
      for (std::size_t i = 0; i < n; ++i) {
        const bench::Args a{t_common.host(i, 0), t_common.host(i, 1), t_common.host(i, 2)};
        if (laplace::math::own::detail_ibx::direct_side(a[0], a[1], a[2]).method == m) v.push_back(a);
      }
      std::printf("method %d: %zu arguments ...", m, v.size());
      std::fflush(stdout);
      bench::eval_kernel(v, Dm<0>{});
      if (mode0 == "seqr" || mode0 == "seqrc" || mode0 == "seqc") {
        const auto h = bench::to_host_args(v);
        bench::DArgs d("d", v.size());
        Kokkos::deep_copy(d, h);
        if (mode0 != "seqc") std::printf(" reduce dm %.3f", bench::time_ns<bench::Exec>(d, Dm<0>{}, 7));
        if (mode0 != "seqr") std::printf(" reduce cf %.3f", bench::time_ns<bench::Exec>(d, Cf<0>{}, 7));
      }
      std::printf(" ok\n");
      std::fflush(stdout);
    }
    return 0;
  }
  if (argc > 2 && std::string(argv[2]).rfind("methods", 0) == 0) {
    const std::string mode = argv[2];
    const auto& t = mode == "methods-mixed" ? t_mixed : mode == "methods-large" ? t_large : t_common;
    // ns per call of dm and cf on one host thread, per method chosen by dm (common distribution)
    std::map<int, std::vector<bench::Args>> by;
    for (std::size_t i = 0; i < n; ++i) {
      const bench::Args a{t.host(i, 0), t.host(i, 1), t.host(i, 2)};
      by[laplace::math::own::detail_ibx::direct_side(a[0], a[1], a[2], g_th).method].push_back(a);
    }
    for (const auto& [m, v] : by) {
      const auto h = bench::to_host_args(v);
      std::printf("method %d: share %.3f | ns/call Serial dm %8.2f, cf %8.2f", m, double(v.size()) / n,
                  bench::time_ns<bench::HostOne>(h, Dm<0>{}, 5), bench::time_ns<bench::HostOne>(h, Cf<0>{}, 5));
      std::fflush(stdout);
      // the same arguments in a kernel: largest relative difference from the host, and the time
      const auto yk = bench::eval_kernel(v, Dm<0>{});
      const auto yh = bench::eval_host(v, Dm<0>{});
      double diff = 0;
      for (std::size_t i = 0; i < v.size(); ++i)
        if (yh[i] != yk[i]) diff = std::max(diff, std::fabs(yk[i] - yh[i]) / std::max(std::fabs(yh[i]), 1e-300));
      bench::DArgs d("d", v.size());
      Kokkos::deep_copy(d, h);
      std::printf(" | %s: dm %7.3f, cf %7.3f | kernel vs host: max relative difference %.3g\n", bench::Exec::name(),
                  bench::time_ns<bench::Exec>(d, Dm<0>{}, 7), bench::time_ns<bench::Exec>(d, Cf<0>{}, 7), diff);
      std::fflush(stdout);
    }
    return 0;
  }
  timing("dm", Dm<0>{});
  timing("cf", Cf<0>{});
  timing("boost", Boost<0>{});
#ifndef KOKKOS_ENABLE_CUDA
  timing("boost-ld", BoostLd<0>{}, false);
#endif
  return 0;
}
