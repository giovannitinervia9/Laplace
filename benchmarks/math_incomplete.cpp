// Accuracy and speed of the incomplete gamma and beta functions, their
// inverses, erfc and the normal quantile: Boost.Math (GPU-capable) and, where
// one exists, the platform's.
//
// Large shapes (gamma_*_large, a from 1e2 to 1e15): the classic route of
// Laplace (the series and the continued fraction, O(sqrt(a)) terms near x = a),
// the Temme route alone, and the chosen dispatch gamma_inc. The classic route is
// measured only where it is affordable (a <= 1e8, about 1e4 terms; beyond that
// it needs up to 1e8 terms per call), and its timing on a <= 1e6. Results are
// also split by region (a and |x/a - 1|).
//
// Usage: math_incomplete <reference dir> <output dir> [name filter]
#include "measure.hpp"

#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/log1p.hpp>

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

// Large shapes: the classic route, the Temme route and the chosen dispatch, each
// as one value of the pair (P or Q, or their logarithms).
KOKKOS_INLINE_FUNCTION double temme_phi(double a, double x) {
  return -boost::math::log1pmx((x - a) / a, boost_policy());
}
struct LargeClassicP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_classic(a, x).p; }
};
struct LargeClassicQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_classic(a, x).q; }
};
struct LargeClassicLogP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_classic(a, x).log_p; }
};
struct LargeClassicLogQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_classic(a, x).log_q; }
};
struct LargeTemmeP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_temme(a, x, temme_phi(a, x)).p; }
};
struct LargeTemmeQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_temme(a, x, temme_phi(a, x)).q; }
};
struct LargeTemmeLogP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_temme(a, x, temme_phi(a, x)).log_p; }
};
struct LargeTemmeLogQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc_temme(a, x, temme_phi(a, x)).log_q; }
};
struct LargeChosenP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc(a, x).p; }
};
struct LargeChosenQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc(a, x).q; }
};
struct LargeChosenLogP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc(a, x).log_p; }
};
struct LargeChosenLogQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::own::gamma_inc(a, x).log_q; }
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

// The reference rows with a <= a_max (to bound the cost of the classic route).
bench::Ref subset_a_at_most(const bench::Ref& r, double a_max) {
  bench::Ref s;
  s.nargs = r.nargs;
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    if (r.x[i][0] > a_max) continue;
    s.x.push_back(r.x[i]);
    s.hi.push_back(r.hi[i]);
    s.lo.push_back(r.lo[i]);
  }
  return s;
}

// Errors split by a in [1e2, 1e4), [1e4, 1e8), [1e8, 1e15] and by |x/a - 1| in
// [0, 2^-20), [2^-20, 2^-10), [2^-10, 2^-5), [2^-5, 1]. One line per cell.
template <class F>
void regions(const char* fn, const char* impl, const bench::Ref& r, F f) {
  struct Cell {
    double ulp = 0, ulp1 = 0, ulpL = 0;
    std::size_t n = 0, bad = 0;
  };
  Cell cell[3][4];
  const auto y = bench::eval_kernel(r.x, f);
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double a = r.x[i][0], x = r.x[i][1];
    const double d = std::fabs(x / a - 1.0);
    const int ia = a < 1e4 ? 0 : (a < 1e8 ? 1 : 2);
    const int id = d < 0x1p-20 ? 0 : (d < 0x1p-10 ? 1 : (d < 0x1p-5 ? 2 : 3));
    Cell& c = cell[ia][id];
    ++c.n;
    const double e = std::fabs(bench::error_in(y[i], r.hi[i], r.lo[i], r.hi[i]));
    const double e1 = std::fabs(bench::error_in(y[i], r.hi[i], r.lo[i], std::max(std::fabs(r.hi[i]), 1.0)));
    if (std::isinf(e)) {
      ++c.bad;
      continue;
    }
    c.ulp = std::max(c.ulp, e);
    c.ulp1 = std::max(c.ulp1, e1);
    if (r.hi[i] > 0) c.ulpL = std::max(c.ulpL, e / std::max(1.0, std::fabs(std::log(r.hi[i]))));
  }
  static const char* alab[3] = {"[1e2, 1e4)", "[1e4, 1e8)", "[1e8, 1e15]"};
  static const char* dlab[4] = {"< 2^-20", "[2^-20, 2^-10)", "[2^-10, 2^-5)", ">= 2^-5"};
  for (int ia = 0; ia < 3; ++ia)
    for (int id = 0; id < 4; ++id) {
      const Cell& c = cell[ia][id];
      if (c.n == 0) continue;
      std::printf("  %-10s %-7s a %-12s |d| %-15s n %5zu | max ulp %9.4g, ulp1 %8.4g, ulpL %8.4g%s\n", fn, impl,
                  alab[ia], dlab[id], c.n, c.ulp, c.ulp1, c.ulpL, c.bad ? " (non-finite errors)" : "");
    }
}

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
  // large shapes, as in the references: a log-uniform in [1e2, 1e15], |d| log-uniform in [2^-40, 1]
  auto large_arg = [&](double amax_exp) {
    const double a = std::pow(10.0, 2.0 + (amax_exp - 2.0) * u01(gen));
    const double d = std::exp2(-40.0 * u01(gen)) * (u01(gen) < 0.5 ? 1.0 : -1.0);
    return bench::Args{a, a * (1.0 + d), 0};
  };
  const auto t_large = bench::make_time_args(n, [&] { return large_arg(15.0); });
  const auto t_large_mid = bench::make_time_args(n, [&] { return large_arg(6.0); });

  auto run = [&](const char* name, const char* impl, auto f, const bench::TimeArgs& t) {
    if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
    bench::report(name, impl, bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t);
  };
  // the classic route is timed and checked on the rows it affords (a <= 1e6 and a <= 1e8)
  auto run_large = [&](const char* name, const char* impl, auto f, const bench::TimeArgs& t, double a_max) {
    if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
    const auto r = subset_a_at_most(bench::read_ref(refdir + "/" + name + ".csv"), a_max);
    bench::report(name, impl, r, f, outdir, t);
    regions(name, impl, r, f);
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

  // large shapes: Boost in double for reference, then the three Laplace routes
  // (Boost is left out here: outside its own Temme zone, a sigma^2 < 20, it
  // sums series of O(1/sigma) terms, about 1e6 at a = 1e15, which made this
  // bench run for hours)
  run_large("gamma_p_large", "classic", LargeClassicP{}, t_large_mid, 1e8);
  run_large("gamma_q_large", "classic", LargeClassicQ{}, t_large_mid, 1e8);
  run_large("log_gamma_p_large", "classic", LargeClassicLogP{}, t_large_mid, 1e8);
  run_large("log_gamma_q_large", "classic", LargeClassicLogQ{}, t_large_mid, 1e8);
  run_large("gamma_p_large", "temme", LargeTemmeP{}, t_large, 1e15);
  run_large("gamma_q_large", "temme", LargeTemmeQ{}, t_large, 1e15);
  run_large("log_gamma_p_large", "temme", LargeTemmeLogP{}, t_large, 1e15);
  run_large("log_gamma_q_large", "temme", LargeTemmeLogQ{}, t_large, 1e15);
  run_large("gamma_p_large", "chosen", LargeChosenP{}, t_large, 1e15);
  run_large("gamma_q_large", "chosen", LargeChosenQ{}, t_large, 1e15);
  run_large("log_gamma_p_large", "chosen", LargeChosenLogP{}, t_large, 1e15);
  run_large("log_gamma_q_large", "chosen", LargeChosenLogQ{}, t_large, 1e15);
  return 0;
}
