// Accuracy and speed of the special functions: platform, Boost.Math, own.
//
// Usage: math_special <reference dir> <output dir>
//
// Boost.Math (1.90, from the R package BH; Boost Software License) is called
// with a policy that never throws and never promotes double to long double,
// so that host and device follow the same code path.
#include "measure.hpp"

#include <laplace/core/config.hpp>

#include <laplace/math/polygamma.hpp>

#include <boost/math/special_functions/digamma.hpp>
#ifndef KOKKOS_ENABLE_CUDA
#include <boost/math/special_functions/polygamma.hpp>  // no GPU version
#endif
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

#include <random>

namespace {

namespace bp = boost::math::policies;
using boost_policy = bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>,
                                bp::overflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>,
                                bp::promote_double<false>>;

struct LgammaPlatform {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Kokkos::lgamma(x); }
};
struct LgammaBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return boost::math::lgamma(x, boost_policy()); }
};
struct DigammaBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return boost::math::digamma(x, boost_policy()); }
};
struct TrigammaBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return boost::math::trigamma(x, boost_policy()); }
};

struct PolygammaOwn {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::own::polygamma(n, x); }
};
#ifndef KOKKOS_ENABLE_CUDA
struct PolygammaBoost {
  int n;
  double operator()(double x) const { return boost::math::polygamma(n, x, boost_policy()); }
};
#endif

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_special <reference dir> <output dir>\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  bench::print_spaces();

  // typical arguments of the distributions: shapes, counts plus one, degrees of freedom
  std::mt19937_64 gen(20261108);
  std::uniform_real_distribution<double> u(-5.0, 15.0);
  const auto t = bench::make_time_args(std::size_t(1) << 23, [&] { return std::exp2(u(gen)); });

  const auto r_lgamma = bench::read_ref(refdir + "/lgamma.csv");
  const auto r_digamma = bench::read_ref(refdir + "/digamma.csv");
  const auto r_trigamma = bench::read_ref(refdir + "/trigamma.csv");
  bench::report("lgamma", "platform", r_lgamma, LgammaPlatform{}, outdir, t);
  bench::report("lgamma", "boost", r_lgamma, LgammaBoost{}, outdir, t);
  bench::report("digamma", "boost", r_digamma, DigammaBoost{}, outdir, t);
  bench::report("trigamma", "boost", r_trigamma, TrigammaBoost{}, outdir, t);
  bench::report("digamma", "own", r_digamma, PolygammaOwn{0}, outdir, t);
  bench::report("trigamma", "own", r_trigamma, PolygammaOwn{1}, outdir, t);
  for (int n = 2; n <= 6; ++n) {
    const std::string name = "polygamma" + std::to_string(n);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    bench::report(name.c_str(), "own", r, PolygammaOwn{n}, outdir, t);
#ifndef KOKKOS_ENABLE_CUDA
    bench::report(name.c_str(), "boost", r, PolygammaBoost{n}, outdir, t, false);
#endif
  }
  return 0;
}
