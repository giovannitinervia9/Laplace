// Accuracy and speed of the incomplete gamma and beta functions, their
// inverses, erfc and the normal quantile: Boost.Math (GPU-capable) and, where
// one exists, the platform's.
//
// Usage: math_incomplete <reference dir> <output dir>
#include "measure.hpp"

#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <random>

namespace {

using laplace::math::detail::boost_policy;

struct GammaP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    return boost::math::gamma_p(a, x, boost_policy());
  }
};
struct GammaQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    return boost::math::gamma_q(a, x, boost_policy());
  }
};
struct GammaPOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_p(a, x); }
};
struct GammaQOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_q(a, x); }
};
struct GammaPInv {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double p) const {
    return boost::math::gamma_p_inv(a, p, boost_policy());
  }
};
struct Ibeta {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return boost::math::ibeta(a, b, x, boost_policy());
  }
};
struct Ibetac {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return boost::math::ibetac(a, b, x, boost_policy());
  }
};
struct IbetaOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return laplace::math::own::beta_inc(a, b, x).i;
  }
};
struct IbetacOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return laplace::math::own::beta_inc(a, b, x).ic;
  }
};
struct IbetaInv {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double p) const {
    return boost::math::ibeta_inv(a, b, p, boost_policy());
  }
};
struct ErfcPlatform {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Kokkos::erfc(x); }
};
struct ErfcBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return boost::math::erfc(x, boost_policy()); }
};
// Phi^-1(p) = -sqrt(2) erfc^-1(2 p); for p >= 1/2 through 1 - p, which is exact
struct QnormBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double p) const {
    constexpr double sqrt2 = 1.41421356237309504880;
    if (p < 0.5) return -sqrt2 * boost::math::erfc_inv(2.0 * p, boost_policy());
    return sqrt2 * boost::math::erfc_inv(2.0 * (1.0 - p), boost_policy());
  }
};

#ifndef KOKKOS_ENABLE_CUDA
// Host only: Boost's default promotion of double to long double (80-bit with gcc).
namespace bp = boost::math::policies;
using promote_policy = bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>,
                                  bp::overflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>>;
struct GammaPPromote {
  double operator()(double a, double x) const { return boost::math::gamma_p(a, x, promote_policy()); }
};
struct GammaQPromote {
  double operator()(double a, double x) const { return boost::math::gamma_q(a, x, promote_policy()); }
};
struct GammaPInvPromote {
  double operator()(double a, double p) const { return boost::math::gamma_p_inv(a, p, promote_policy()); }
};
struct IbetaPromote {
  double operator()(double a, double b, double x) const { return boost::math::ibeta(a, b, x, promote_policy()); }
};
struct IbetacPromote {
  double operator()(double a, double b, double x) const { return boost::math::ibetac(a, b, x, promote_policy()); }
};
struct IbetaInvPromote {
  double operator()(double a, double b, double p) const { return boost::math::ibeta_inv(a, b, p, promote_policy()); }
};
#endif

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_incomplete <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";  // run only the functions whose name contains it
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 21;
  std::mt19937_64 gen(20261108);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> z(0.0, 1.0);
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };

  const auto t_gamma = bench::make_time_args(n, [&] {
    const double a = expo(-3, 10);
    return bench::Args{a, a * std::exp(0.5 * z(gen)), 0};
  });
  const auto t_gamma_inv = bench::make_time_args(n, [&] { return bench::Args{expo(-3, 10), u01(gen), 0}; });
  const auto t_beta = bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), u01(gen)}; });
  const auto t_erfc = bench::make_time_args(n, [&] { return -5.0 + 15.0 * u01(gen); });
  const auto t_p = bench::make_time_args(n, [&] { return u01(gen); });

  auto run = [&](const char* name, const char* impl, auto f, const bench::TimeArgs& t) {
    if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
    bench::report(name, impl, bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t);
  };
  run("gamma_p", "boost", GammaP{}, t_gamma);
  run("gamma_q", "boost", GammaQ{}, t_gamma);
  run("gamma_p", "own", GammaPOwn{}, t_gamma);
  run("gamma_q", "own", GammaQOwn{}, t_gamma);
  run("gamma_p_inv", "boost", GammaPInv{}, t_gamma_inv);
  run("ibeta", "boost", Ibeta{}, t_beta);
  run("ibetac", "boost", Ibetac{}, t_beta);
  run("ibeta", "own", IbetaOwn{}, t_beta);
  run("ibetac", "own", IbetacOwn{}, t_beta);
  run("ibeta_inv", "boost", IbetaInv{}, t_beta);
#ifndef KOKKOS_ENABLE_CUDA
  auto run_host = [&](const char* name, const char* impl, auto f, const bench::TimeArgs& t) {
    if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
    bench::report(name, impl, bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t, false);
  };
  run_host("gamma_p", "boost-ld", GammaPPromote{}, t_gamma);
  run_host("gamma_q", "boost-ld", GammaQPromote{}, t_gamma);
  run_host("gamma_p_inv", "boost-ld", GammaPInvPromote{}, t_gamma_inv);
  run_host("ibeta", "boost-ld", IbetaPromote{}, t_beta);
  run_host("ibetac", "boost-ld", IbetacPromote{}, t_beta);
  run_host("ibeta_inv", "boost-ld", IbetaInvPromote{}, t_beta);
#endif
  run("erfc", "platform", ErfcPlatform{}, t_erfc);
  run("erfc", "boost", ErfcBoost{}, t_erfc);
  run("qnorm", "boost", QnormBoost{}, t_p);
  return 0;
}
