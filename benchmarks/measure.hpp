// Shared machinery of the measurement tools and of the accuracy tests:
// reference files, errors in ulps, evaluation in a Kokkos kernel and on the
// host, timing.
//
// Reference files (tests/data/reference/<fn>.csv, written by tools/reference/*.py)
// hold one to three arguments followed by hi, lo, all in hexadecimal; hi + lo
// is the exact value as a double-double. A candidate is a functor callable
// with as many doubles as the file has arguments.
// Two errors are reported:
//   ulp      ((y - hi) - lo) / ulp(hi): the relative error in the last place,
//            large near a zero of the function;
//   ulp1     ((y - hi) - lo) / ulp(max(|hi|, 1)): the absolute error in units
//            of ulp(1) when |f| < 1, which is what matters when f is added to
//            quantities of order one (a log-density, a score).
#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace bench {

using Exec = Kokkos::DefaultExecutionSpace;
using HostOne = Kokkos::Serial;  // one host thread
using Args = std::array<double, 3>;
using DArgs = Kokkos::View<double* [3], Kokkos::LayoutRight, Exec::memory_space>;
using HArgs = Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace>;

struct Ref {
  int nargs = 1;
  std::vector<Args> x;
  std::vector<double> hi, lo;
};

inline Ref read_ref(const std::string& path) {
  Ref r;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::string line;
  std::getline(in, line);  // header: arguments, then hi, lo
  r.nargs = static_cast<int>(std::count(line.begin(), line.end(), ',')) - 1;
  while (std::getline(in, line)) {
    char* p = line.data();
    char* end = nullptr;
    Args a{0.0, 0.0, 0.0};
    for (int k = 0; k < r.nargs; ++k) {
      a[k] = std::strtod(p, &end);
      p = end + 1;
    }
    r.x.push_back(a);
    r.hi.push_back(std::strtod(p, &end));
    r.lo.push_back(std::strtod(end + 1, &end));
  }
  return r;
}

// call f with its arity
template <class F>
KOKKOS_INLINE_FUNCTION double call(const F& f, double a0, double a1, double a2) {
  if constexpr (std::is_invocable_v<F, double>) {
    return f(a0);
  } else if constexpr (std::is_invocable_v<F, double, double>) {
    return f(a0, a1);
  } else {
    return f(a0, a1, a2);
  }
}

inline double ulp_of(double a) {
  a = std::fabs(a);
  const int e = a < 2.2250738585072014e-308 ? -1022 : std::ilogb(a);
  return std::ldexp(1.0, e - 52);
}

// error in ulps of `scale`; infinities and NaNs must match exactly
inline double error_in(double y, double hi, double lo, double scale) {
  if (std::isnan(hi)) return std::isnan(y) ? 0.0 : INFINITY;
  if (std::isinf(hi)) return y == hi ? 0.0 : INFINITY;
  if (std::isnan(y) || std::isinf(y)) return INFINITY;
  return ((y - hi) - lo) / ulp_of(scale);
}

struct Accuracy {
  double max_ulp = 0, mean_ulp = 0, frac_cr = 0, max_ulp1 = 0;
  double max_ulpL = 0;  // ulp / max(1, |log f|): beyond the inherent error of exp(-|log f|)
  Args worst{0, 0, 0};
  std::size_t n_bad = 0;  // wrong infinity or NaN (excluded from the other figures)
  Args bad{0, 0, 0};
};

inline Accuracy accuracy(const Ref& r, const std::vector<double>& y) {
  Accuracy a;
  std::size_t cr = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double e = std::fabs(error_in(y[i], r.hi[i], r.lo[i], r.hi[i]));
    const double e1 = std::fabs(error_in(y[i], r.hi[i], r.lo[i], std::max(std::fabs(r.hi[i]), 1.0)));
    if (std::isinf(e)) {
      if (a.n_bad++ == 0) a.bad = r.x[i];
      continue;
    }
    if (e > a.max_ulp) {
      a.max_ulp = e;
      a.worst = r.x[i];
    }
    a.max_ulp1 = std::max(a.max_ulp1, e1);
    if (r.hi[i] > 0) a.max_ulpL = std::max(a.max_ulpL, e / std::max(1.0, std::fabs(std::log(r.hi[i]))));
    a.mean_ulp += e;
    if (e <= 0.5) ++cr;
  }
  const std::size_t good = y.size() - a.n_bad;
  if (good > 0) {
    a.mean_ulp /= static_cast<double>(good);
    a.frac_cr = static_cast<double>(cr) / static_cast<double>(good);
  }
  return a;
}

inline HArgs to_host_args(const std::vector<Args>& x) {
  HArgs h("args", x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    for (int k = 0; k < 3; ++k) h(i, k) = x[i][k];
  return h;
}

template <class F>
std::vector<double> eval_kernel(const std::vector<Args>& x, F f) {
  const std::size_t n = x.size();
  const auto hx = to_host_args(x);
  DArgs dx("x", n);
  Kokkos::deep_copy(dx, hx);
  Kokkos::View<double*, Exec::memory_space> dy("y", n);
  Kokkos::parallel_for(
      "eval", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) { dy(i) = call(f, dx(i, 0), dx(i, 1), dx(i, 2)); });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = hy(i);
  return y;
}

template <class F>
std::vector<double> eval_host(const std::vector<Args>& x, F f) {
  std::vector<double> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = call(f, x[i][0], x[i][1], x[i][2]);
  return y;
}

template <class Space, class F>
double time_ns(const Kokkos::View<double* [3], Kokkos::LayoutRight, typename Space::memory_space>& x, F f,
               int reps) {
  const std::size_t n = x.extent(0);
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    double sum = 0;
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_reduce(
        "time", Kokkos::RangePolicy<Space>(0, n),
        KOKKOS_LAMBDA(const std::size_t i, double& acc) { acc += call(f, x(i, 0), x(i, 1), x(i, 2)); }, sum);
    Kokkos::fence();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sum == 12345.678) std::printf(" ");  // keep the sum alive
    best = std::min(best, s);
  }
  return best * 1e9 / static_cast<double>(n);
}

inline void write_bits(const std::string& path, const std::vector<double>& y) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fwrite(y.data(), sizeof(double), y.size(), f);
  std::fclose(f);
}

// Timing arguments, on the host and copied to the default execution space.
struct TimeArgs {
  DArgs dev;
  HArgs host;
};

// draw() returns a double (one argument) or an Args
template <class Draw>
TimeArgs make_time_args(std::size_t n, Draw draw) {
  TimeArgs t{DArgs("td", n), HArgs("th", n)};
  for (std::size_t i = 0; i < n; ++i) {
    Args a{0, 0, 0};
    if constexpr (std::is_same_v<decltype(draw()), double>) {
      a[0] = draw();
    } else {
      a = draw();
    }
    for (int k = 0; k < 3; ++k) t.host(i, k) = a[k];
  }
  Kokkos::deep_copy(t.dev, t.host);
  return t;
}

inline std::string show(const Args& a, int nargs) {
  char buf[96];
  if (nargs == 1) std::snprintf(buf, sizeof buf, "%a", a[0]);
  else if (nargs == 2) std::snprintf(buf, sizeof buf, "%.6g, %.6g", a[0], a[1]);
  else std::snprintf(buf, sizeof buf, "%.6g, %.6g, %.6g", a[0], a[1], a[2]);
  return buf;
}

// One line per (function, implementation). `device` false: the candidate is
// host-only (evaluated and timed on the host only).
template <class F>
void report(const char* fn, const char* impl, const Ref& r, F f, const std::string& outdir,
            const TimeArgs& t, bool device = true) {
  const auto yh = eval_host(r.x, f);
  const auto yk = device ? eval_kernel(r.x, f) : yh;
  const Accuracy ak = accuracy(r, yk), ah = accuracy(r, yh);
  write_bits(outdir + "/" + fn + "_" + impl + ".bin", yk);
  const double t_host1 = time_ns<HostOne>(t.host, f, 5);
  char dev[64] = "-";
  if (device) std::snprintf(dev, sizeof dev, "%.4f", time_ns<Exec>(t.dev, f, 7));
  std::printf("%-10s %-9s | %s: max %9.4g ulp, %8.4g ulp1, %8.4g ulpL, mean %.3f, CR %.4f, worst (%s) | host max %9.4g ulp"
              " | ns/call Serial %7.3f, %s %s | non-finite errors %zu%s%s%s\n",
              fn, impl, device ? Exec::name() : "host", ak.max_ulp, ak.max_ulp1, ak.max_ulpL, ak.mean_ulp, ak.frac_cr,
              show(ak.worst, r.nargs).c_str(), ah.max_ulp, t_host1, Exec::name(), dev, ak.n_bad,
              ak.n_bad ? " (first at " : "", ak.n_bad ? show(ak.bad, r.nargs).c_str() : "", ak.n_bad ? ")" : "");
}

inline void print_spaces() {
  std::printf("default execution space %s (concurrency %d)\n", Exec::name(), Exec().concurrency());
}

}  // namespace bench
