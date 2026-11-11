// Accuracy and speed of the double-double sqrt, exp and log
// (include/laplace/math/detail/double_double.hpp) against the mpmath references
// of tools/reference/double_double.py (tests/data/reference/dd_<fn>.csv).
//
// Usage: math_dd <reference dir>
//
// Accuracy: every reference row is evaluated in a Kokkos kernel on the default
// execution space, which returns the two parts (hi, lo) of the result. Errors
// are in units of 2^-104 |f| ("ddulp"), with f the reference value hi + lo;
// for log also in units of 2^-104 max(|f|, 1) ("ddulp1"), the absolute measure
// that matters near log 1 = 0. The error of a result y = yh + yl is
// ((yh - hi) - lo) + yl, which is exact to double precision because yh and hi
// agree to about 2^-52. Results below 2^-1022 in magnitude are subnormal, where
// the unit 2^-104 |f| is meaningless: they are counted apart and measured in
// units of 2^-1074.
//
// Speed: the ns per call of the double-double function and of the plain double
// function (sqrt, exp, log of the platform), on one host thread (Serial), on
// the default execution space. The double-double calls return hi + lo so that
// no call is removed.
#include <Kokkos_Core.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/math/elementary.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "measure.hpp"

namespace {

namespace ddm = laplace::math::detail::ddm;
using ddm::dd;

constexpr double kTwoM104 = 0x1p-104;
constexpr double kTwoM1022 = 0x1p-1022;
constexpr double kTwoM1074 = 0x1p-1074;

// the three double-double functions, as functors of the argument's two parts
struct DdSqrt {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::sqrt(dd(h, l)); }
};
struct DdExp {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::exp(dd(h, l)); }
};
struct DdLog {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::log(dd(h, l)); }
};
// the same, returning one double (hi + lo) for the timing
template <class F>
struct Sum2 {
  F f;
  KOKKOS_INLINE_FUNCTION double operator()(double h, double l) const {
    const dd r = f(h, l);
    return r.hi + r.lo;
  }
};

// the plain double functions of the platform, for the cost comparison
struct PlainSqrt {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Kokkos::sqrt(x); }
};
struct PlainExp {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::exp(x); }
};
struct PlainLog {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::log(x); }
};

struct Err {
  bool bad = false;     // wrong infinity or NaN
  bool normal = false;  // |f| >= 2^-1022
  double rel = 0;       // ddulp
  double rel1 = 0;      // ddulp1
  double sub = 0;       // subnormal or zero reference: error in units of 2^-1074
};

Err dd_error(const dd& y, double hi, double lo) {
  Err e;
  if (std::isnan(hi) || std::isinf(hi)) {
    e.bad = !(y.hi == hi || (std::isnan(hi) && std::isnan(y.hi)));
    return e;
  }
  if (std::isnan(y.hi) || std::isinf(y.hi)) {
    e.bad = true;
    return e;
  }
  const double diff = ((y.hi - hi) - lo) + y.lo;
  const double ah = std::fabs(hi);
  if (ah >= kTwoM1022) {
    e.normal = true;
    e.rel = std::fabs(diff) / (kTwoM104 * ah);
    e.rel1 = std::fabs(diff) / (kTwoM104 * std::max(ah, 1.0));
  } else {
    e.sub = std::fabs(diff) / kTwoM1074;
  }
  return e;
}

struct Summary {
  double max_rel = 0, sum_rel = 0, max_rel1 = 0, max_sub = 0;
  std::size_t n = 0, n_normal = 0, n_sub = 0, n_bad = 0, n_within1 = 0;
  bench::Args worst{0, 0, 0}, worst1{0, 0, 0}, bad{0, 0, 0};
};

template <class F>
std::vector<dd> eval_dd(const std::vector<bench::Args>& x, F f) {
  const std::size_t n = x.size();
  Kokkos::View<double* [2], Kokkos::LayoutRight, bench::Exec::memory_space> dx("x", n), dy("y", n);
  auto hx = Kokkos::create_mirror_view(dx);
  for (std::size_t i = 0; i < n; ++i) {
    hx(i, 0) = x[i][0];
    hx(i, 1) = x[i][1];
  }
  Kokkos::deep_copy(dx, hx);
  Kokkos::parallel_for(
      "eval_dd", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t i) {
        const dd r = f(dx(i, 0), dx(i, 1));
        dy(i, 0) = r.hi;
        dy(i, 1) = r.lo;
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<dd> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = dd(hy(i, 0), hy(i, 1));
  return y;
}

Summary summarize(const bench::Ref& r, const std::vector<dd>& y) {
  Summary s;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const Err e = dd_error(y[i], r.hi[i], r.lo[i]);
    if (e.bad) {
      if (s.n_bad++ == 0) s.bad = r.x[i];
      continue;
    }
    ++s.n;
    if (!e.normal) {
      ++s.n_sub;
      s.max_sub = std::max(s.max_sub, e.sub);
      continue;
    }
    ++s.n_normal;
    if (e.rel > s.max_rel) {
      s.max_rel = e.rel;
      s.worst = r.x[i];
    }
    if (e.rel1 > s.max_rel1) {
      s.max_rel1 = e.rel1;
      s.worst1 = r.x[i];
    }
    s.sum_rel += e.rel;
    if (e.rel <= 1.0) ++s.n_within1;
  }
  return s;
}

void report(const char* fn, const bench::Ref& r, const Summary& s, bool with_rel1) {
  std::printf("%-6s %zu rows (%zu normal, %zu subnormal or zero): max %8.4g ddulp (x_hi %s), mean %.4g, "
              "fraction <= 1 ddulp %.4f\n",
              fn, r.x.size(), s.n_normal, s.n_sub, s.max_rel, bench::show(s.worst, r.nargs).c_str(),
              s.n_normal ? s.sum_rel / static_cast<double>(s.n_normal) : 0.0,
              s.n_normal ? static_cast<double>(s.n_within1) / static_cast<double>(s.n_normal) : 0.0);
  if (with_rel1)
    std::printf("       max %8.4g ddulp1 (x_hi %s)\n", s.max_rel1, bench::show(s.worst1, r.nargs).c_str());
  std::printf("       subnormal or zero results: max %.4g units of 2^-1074 | non-finite errors %zu%s%s%s\n",
              s.max_sub, s.n_bad, s.n_bad ? " (first at x_hi " : "", s.n_bad ? bench::show(s.bad, r.nargs).c_str() : "",
              s.n_bad ? ")" : "");
}

// arguments for the timings: x_hi uniform in [a, b], x_lo = 0
bench::TimeArgs dd_args(std::size_t n, double a, double b, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(a, b);
  return bench::make_time_args(n, [&] { return bench::Args{u(rng), 0.0, 0.0}; });
}
bench::TimeArgs plain_args(std::size_t n, double a, double b, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(a, b);
  return bench::make_time_args(n, [&] { return u(rng); });
}

void timing(const char* fn, const bench::TimeArgs& td, const bench::TimeArgs& tp, auto dd_f, auto plain_f) {
  const double dd_host = bench::time_ns<bench::HostOne>(td.host, dd_f, 5);
  const double dd_dev = bench::time_ns<bench::Exec>(td.dev, dd_f, 7);
  const double pl_host = bench::time_ns<bench::HostOne>(tp.host, plain_f, 5);
  const double pl_dev = bench::time_ns<bench::Exec>(tp.dev, plain_f, 7);
  std::printf("%-6s ns/call  double-double: Serial %8.3f, %s %8.3f | plain double: Serial %7.3f, %s %7.3f\n", fn,
              dd_host, bench::Exec::name(), dd_dev, pl_host, bench::Exec::name(), pl_dev);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_dd <reference dir>\n");
    return 2;
  }
  bench::print_spaces();
  const std::string dir = argv[1];
  const std::size_t n_time = 1 << 20;

  {
    const auto r = bench::read_ref(dir + "/dd_sqrt.csv");
    report("sqrt", r, summarize(r, eval_dd(r.x, DdSqrt{})), false);
  }
  {
    const auto r = bench::read_ref(dir + "/dd_exp.csv");
    report("exp", r, summarize(r, eval_dd(r.x, DdExp{})), false);
  }
  {
    const auto r = bench::read_ref(dir + "/dd_log.csv");
    report("log", r, summarize(r, eval_dd(r.x, DdLog{})), true);
  }

  std::printf("\nspeed on %zu arguments, minimum over repetitions\n", n_time);
  timing("sqrt", dd_args(n_time, 0.5, 100.0, 1), plain_args(n_time, 0.5, 100.0, 1),
         Sum2<DdSqrt>{DdSqrt{}}, PlainSqrt{});
  timing("exp", dd_args(n_time, -20.0, 20.0, 2), plain_args(n_time, -20.0, 20.0, 2),
         Sum2<DdExp>{DdExp{}}, PlainExp{});
  timing("log", dd_args(n_time, 0.5, 100.0, 3), plain_args(n_time, 0.5, 100.0, 3),
         Sum2<DdLog>{DdLog{}}, PlainLog{});
  return 0;
}
