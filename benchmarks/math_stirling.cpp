// Candidates for Stirling's error delta(a) and its derivatives to order 6
// (include/laplace/math/stirling.hpp): the definition (direct), the upward
// recurrence, the asymptotic series and the chosen routes, by order and by region of a.
// ns per call on one host thread (Serial) and on the default execution space;
// errors against the mpmath references (tools/reference/stirling.py).
//
//   math_stirling <reference dir>
#include "measure.hpp"

#include <laplace/math/stirling.hpp>

#include <cstdio>
#include <random>
#include <string>

namespace lm = laplace::math;
namespace ds = laplace::math::detail_stirling;

namespace {

template <int K>
struct Cand {
  int route;  // 0 direct, 1 recurrence, 2 asymptotic, 3 the chosen routes
  KOKKOS_INLINE_FUNCTION double operator()(double a) const {
    return route == 0 ? ds::direct<K>(a) : route == 1 ? ds::recurrence<K>(a) : route == 2 ? ds::asymptotic<K>(a) : ds::value<K>(a);
  }
};

bench::Ref restrict(const bench::Ref& r, double lo, double hi) {
  bench::Ref o;
  o.nargs = 1;
  for (std::size_t i = 0; i < r.x.size(); ++i)
    if (r.x[i][0] > lo && r.x[i][0] <= hi) o.x.push_back(r.x[i]), o.hi.push_back(r.hi[i]), o.lo.push_back(r.lo[i]);
  return o;
}

template <class F>
void line(const char* fn, int k, const bench::Ref& r, F f, double lo, double hi, bool timing) {
  const auto rr = restrict(r, lo, hi);
  if (rr.x.empty()) return;
  const auto y = bench::eval_kernel(rr.x, f);
  const auto a = bench::accuracy(rr, y);
  std::printf("%-10s k=%d (%9.3g, %9.3g] rows %4zu  max %9.3g ulp at %-12.6g", fn, k, lo, hi, rr.x.size(), a.max_ulp,
              a.worst[0]);
  if (timing) {
    std::mt19937_64 g(20261112);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const auto t = bench::make_time_args(1 << 14, [&] { return lo * std::pow(hi / lo, u(g)); });
    std::printf(" | ns/call Serial %8.2f, %s %8.3f", bench::time_ns<bench::HostOne>(t.host, f, 3), bench::Exec::name(),
                bench::time_ns<bench::Exec>(t.dev, f, 5));
  }
  std::printf("\n");
}

template <int K>
void order(const std::string& dir) {
  const auto r = bench::read_ref(dir + "/stirling_error" + (K ? "_d" + std::to_string(K) : "") + ".csv");
  const double cuts[] = {1e-12, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 40.0, 1e15};
  const char* names[] = {"direct", "recurrence", "asymptotic", "chosen"};
  for (int route = 0; route < 4; ++route)
    for (int i = 0; i + 1 < 13; ++i) line(names[route], K, r, Cand<K>{route}, cuts[i], cuts[i + 1], i % 3 == 0);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const std::string dir = argc > 1 ? argv[1] : "tests/data/reference";
    bench::print_spaces();
    order<0>(dir);
    order<1>(dir);
    order<2>(dir);
    order<3>(dir);
    order<4>(dir);
    order<5>(dir);
    order<6>(dir);
  }
  Kokkos::finalize();
}
