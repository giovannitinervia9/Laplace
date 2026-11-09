// Accuracy and speed of the elementary functions, platform against own.
//
// Usage: math_elementary <reference dir> <output dir>
//
// Accuracy: every reference argument (tests/data/reference/<fn>.csv, written by
// tools/reference/elementary.py) is evaluated in a Kokkos kernel on the default
// execution space (OpenMP in the CPU build, CUDA in the GPU build) and, for
// comparison, by a plain host loop. The error in units in the last place is
// ((y - hi) - lo) / ulp(hi), hi + lo being the double-double reference. The
// kernel results are also written to <output dir>/<fn>_<impl>.bin so that two
// builds can be compared bit for bit (tools/compare_bits.py).
//
// Speed: n arguments in a typical range, summed in a parallel_reduce so the
// calls cannot be removed; the minimum over repetitions, in nanoseconds per
// call, on one host thread (Serial or the host space with one thread) and on
// the default execution space.
#include <Kokkos_Core.hpp>
#include <laplace/math/elementary.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using HostOne = Kokkos::Serial;  // one host thread
using DView = Kokkos::View<double*, Exec::memory_space>;

struct Ref {
  std::vector<double> x, hi, lo;
};

Ref read_ref(const std::string& path) {
  Ref r;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    char* p = line.data();
    char* end = nullptr;
    const double x = std::strtod(p, &end);
    const double hi = std::strtod(end + 1, &end);
    const double lo = std::strtod(end + 1, &end);
    r.x.push_back(x);
    r.hi.push_back(hi);
    r.lo.push_back(lo);
  }
  return r;
}

// error of y in ulps of the reference hi + lo
double ulp_error(double y, double hi, double lo) {
  if (std::isnan(hi)) return std::isnan(y) ? 0.0 : INFINITY;
  if (std::isinf(hi)) return y == hi ? 0.0 : INFINITY;
  if (std::isnan(y) || std::isinf(y)) return INFINITY;
  const double ahi = std::fabs(hi);
  int e = ahi < 2.2250738585072014e-308 ? -1022 : std::ilogb(ahi);
  const double ulp = std::ldexp(1.0, e - 52);
  return ((y - hi) - lo) / ulp;
}

struct Accuracy {
  double max_abs = 0, mean_abs = 0, frac_cr = 0, max_normal = 0;
  double worst_x = 0;
};

Accuracy accuracy(const Ref& r, const std::vector<double>& y) {
  Accuracy a;
  std::size_t cr = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double e = std::fabs(ulp_error(y[i], r.hi[i], r.lo[i]));
    if (e > a.max_abs) {
      a.max_abs = e;
      a.worst_x = r.x[i];
    }
    if (std::fabs(r.hi[i]) >= 2.2250738585072014e-308 && e > a.max_normal) a.max_normal = e;
    a.mean_abs += e;
    if (e <= 0.5) ++cr;
  }
  a.mean_abs /= static_cast<double>(y.size());
  a.frac_cr = static_cast<double>(cr) / static_cast<double>(y.size());
  return a;
}

template <class F>
std::vector<double> eval_kernel(const std::vector<double>& x, F f) {
  const std::size_t n = x.size();
  DView dx("x", n), dy("y", n);
  auto hx = Kokkos::create_mirror_view(dx);
  for (std::size_t i = 0; i < n; ++i) hx(i) = x[i];
  Kokkos::deep_copy(dx, hx);
  Kokkos::parallel_for(
      "eval", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const std::size_t i) { dy(i) = f(dx(i)); });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = hy(i);
  return y;
}

template <class F>
std::vector<double> eval_host(const std::vector<double>& x, F f) {
  std::vector<double> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = f(x[i]);
  return y;
}

template <class Space, class F>
double time_ns(const Kokkos::View<double*, typename Space::memory_space>& x, F f, int reps) {
  const std::size_t n = x.extent(0);
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    double sum = 0;
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_reduce(
        "time", Kokkos::RangePolicy<Space>(0, n),
        KOKKOS_LAMBDA(const std::size_t i, double& acc) { acc += f(x(i)); }, sum);
    Kokkos::fence();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sum == 12345.678) std::printf(" ");  // keep sum alive
    best = std::min(best, s);
  }
  return best * 1e9 / static_cast<double>(n);
}

void write_bits(const std::string& path, const std::vector<double>& y) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fwrite(y.data(), sizeof(double), y.size(), f);
  std::fclose(f);
}

struct ExpPlatform {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::platform::exp(x); }
};
struct ExpOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::own::exp(x); }
};
struct LogPlatform {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::platform::log(x); }
};
struct LogOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::own::log(x); }
};

template <class F>
void report(const char* fn, const char* impl, const Ref& r, F f, const std::string& outdir,
            const Kokkos::View<double*, Exec::memory_space>& tx,
            const Kokkos::View<double*, Kokkos::HostSpace>& tx_host) {
  const auto yk = eval_kernel(r.x, f);
  const auto yh = eval_host(r.x, f);
  const Accuracy ak = accuracy(r, yk), ah = accuracy(r, yh);
  std::size_t differ = 0;
  for (std::size_t i = 0; i < yk.size(); ++i)
    if (std::memcmp(&yk[i], &yh[i], sizeof(double)) != 0) ++differ;
  write_bits(outdir + "/" + fn + "_" + impl + ".bin", yk);
  const double t_host1 = time_ns<HostOne>(tx_host, f, 5);
  const double t_dev = time_ns<Exec>(tx, f, 7);
  std::printf("%-4s %-9s | kernel: max %8.4f ulp (normal results %.4f), mean %.4f, CR %.5f, worst x = %a | host loop: max %8.4f"
              " | kernel vs host bits differ: %zu | ns/call Serial %.3f, %s %.4f\n",
              fn, impl, ak.max_abs, ak.max_normal, ak.mean_abs, ak.frac_cr, ak.worst_x, ah.max_abs, differ, t_host1,
              Exec::name(), t_dev);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_elementary <reference dir> <output dir>\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  std::printf("default execution space %s (concurrency %d), host space %s (concurrency %d)\n", Exec::name(),
              Exec().concurrency(), Kokkos::DefaultHostExecutionSpace::name(),
              Kokkos::DefaultHostExecutionSpace().concurrency());

  const std::size_t n_time = std::size_t(1) << 23;
  std::mt19937_64 gen(20261108);
  auto make_times = [&](auto draw) {
    Kokkos::View<double*, Kokkos::HostSpace> h("th", n_time);
    for (std::size_t i = 0; i < n_time; ++i) h(i) = draw();
    Kokkos::View<double*, Exec::memory_space> d("td", n_time);
    Kokkos::deep_copy(d, h);
    return std::make_pair(d, h);
  };
  std::uniform_real_distribution<double> u_exp(-700.0, 700.0), u_e(-1000.0, 1000.0);
  auto [tx_exp, tx_exp_h] = make_times([&] { return u_exp(gen); });
  auto [tx_log, tx_log_h] = make_times([&] { return std::exp2(u_e(gen)); });

  const Ref rexp = read_ref(refdir + "/exp.csv");
  const Ref rlog = read_ref(refdir + "/log.csv");
  report("exp", "platform", rexp, ExpPlatform{}, outdir, tx_exp, tx_exp_h);
  report("exp", "own", rexp, ExpOwn{}, outdir, tx_exp, tx_exp_h);
  report("log", "platform", rlog, LogPlatform{}, outdir, tx_log, tx_log_h);
  report("log", "own", rlog, LogOwn{}, outdir, tx_log, tx_log_h);
  return 0;
}
