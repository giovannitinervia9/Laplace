// Speed and accuracy of exprel and log1prel with their derivatives
// (include/laplace/math/exprel.hpp, log1prel.hpp), by order and by region of x,
// and the two candidates for log1prel above x = 6 (the forward recursion and
// the quadrature split at t = 1/4). ns per call on one host thread
// (Serial) and on the default execution space; errors against the mpmath
// references (tools/reference/exprel.py, log1prel.py) restricted to the region.
//
//   math_rel <reference dir>
#include "measure.hpp"

#include <laplace/math/exprel.hpp>
#include <laplace/math/log1prel.hpp>

#include <cstdio>
#include <random>
#include <string>

namespace lm = laplace::math;

namespace {

struct Exprel {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (k) {
      case 0: return lm::exprel(x);
      case 1: return lm::exprel_d1(x);
      case 2: return lm::exprel_d2(x);
      case 3: return lm::exprel_d3(x);
      case 4: return lm::exprel_d4(x);
      case 5: return lm::exprel_d5(x);
      default: return lm::exprel_d6(x);
    }
  }
};

struct Log1prel {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (k) {
      case 0: return lm::log1prel(x);
      case 1: return lm::log1prel_d1(x);
      case 2: return lm::log1prel_d2(x);
      case 3: return lm::log1prel_d3(x);
      case 4: return lm::log1prel_d4(x);
      case 5: return lm::log1prel_d5(x);
      default: return lm::log1prel_d6(x);
    }
  }
};

// the two candidates above x = 6, order k
struct Above6 {
  int k;
  bool quad;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    namespace d = lm::detail_log1prel;
    switch (k) {
      case 1: return quad ? d::quadrature_split<1>(x) : d::recursion<1>(x);
      case 2: return quad ? d::quadrature_split<2>(x) : d::recursion<2>(x);
      case 3: return quad ? d::quadrature_split<3>(x) : d::recursion<3>(x);
      case 4: return quad ? d::quadrature_split<4>(x) : d::recursion<4>(x);
      case 5: return quad ? d::quadrature_split<5>(x) : d::recursion<5>(x);
      default: return quad ? d::quadrature_split<6>(x) : d::recursion<6>(x);
    }
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
void line(const char* fn, int k, const char* region, const bench::Ref& r, F f, double lo, double hi) {
  const auto rr = restrict(r, lo, hi);
  const auto y = bench::eval_kernel(rr.x, f);
  const auto a = bench::accuracy(rr, y);
  std::mt19937_64 g(20261112);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const bool logscale = hi / std::max(lo, 1e-300) > 100.0 && lo > 0;
  const auto t = bench::make_time_args(1 << 16, [&] {
    const double v = u(g);
    return logscale ? lo * std::pow(hi / lo, v) : lo + (hi - lo) * v;
  });
  const double ns1 = bench::time_ns<bench::HostOne>(t.host, f, 5);
  const double nsd = bench::time_ns<bench::Exec>(t.dev, f, 7);
  std::printf("%-9s k=%d %-22s rows %4zu  max %6.3g ulp | ns/call Serial %7.2f, %s %7.3f\n", fn, k, region,
              rr.x.size(), a.max_ulp, ns1, bench::Exec::name(), nsd);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const std::string dir = argc > 1 ? argv[1] : "tests/data/reference";
    bench::print_spaces();
    for (int k = 0; k <= 6; ++k) {
      const auto r = bench::read_ref(dir + "/exprel" + (k ? "_d" + std::to_string(k) : "") + ".csv");
      line("exprel", k, "[-1, 1]", r, Exprel{k}, -1.0, 1.0);
      line("exprel", k, "(1, 30]", r, Exprel{k}, 1.0, 30.0);
      line("exprel", k, "[-30, -1)", r, Exprel{k}, -30.0, -1.0 - 1e-300);
    }
    for (int k = 0; k <= 6; ++k) {
      const auto r = bench::read_ref(dir + "/log1prel" + (k ? "_d" + std::to_string(k) : "") + ".csv");
      line("log1prel", k, "(-1/4, 1/4] series", r, Log1prel{k}, -0.25, 0.25);
      line("log1prel", k, "(1/4, 6] quadrature", r, Log1prel{k}, 0.25, 6.0);
      line("log1prel", k, "(6, 1e5] current", r, Log1prel{k}, 6.0, 1e5);
      if (k > 0) {
        line("recursion", k, "(6, 20]", r, Above6{k, false}, 6.0, 20.0);
        line("quad_split", k, "(6, 20]", r, Above6{k, true}, 6.0, 20.0);
        line("recursion", k, "(20, 1e5]", r, Above6{k, false}, 20.0, 1e5);
      }
    }
  }
  Kokkos::finalize();
}
