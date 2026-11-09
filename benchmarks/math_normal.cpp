// Accuracy and speed of the functions of the normal block (include/laplace/math/normal.hpp):
// erfcx, Phi, log Phi, the derivatives of log Phi to order five, and the
// quantile from p and from log p. Candidates: Laplace's own, Boost.Math, the
// platform (and on CUDA its erfcx, normcdf, normcdfinv), the naive formulas,
// and a port of statmodels7's derivatives of log Phi (pt_logphi.h).
//
// Usage: math_normal <reference dir> <output dir> [name filter]
#include "measure.hpp"

#include <laplace/math/normal.hpp>

#include <boost/math/special_functions/erf.hpp>

#include <random>

namespace {

namespace own = laplace::math::own;
using laplace::math::detail::boost_policy;
constexpr double sqrt_half = 0.70710678118654752440;
constexpr double sqrt2 = 1.41421356237309504880;

// ---------------------------------------------------------------- erfc, erfcx
struct ErfcOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return own::erfc(x); }
};
struct ErfcPlatform {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Kokkos::erfc(x); }
};
struct ErfcBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return boost::math::erfc(x, boost_policy()); }
};
struct ErfcxOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return own::erfcx(x); }
};
struct ErfcxNaive {  // exp(x^2) erfc(x), platform
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Kokkos::exp(x * x) * Kokkos::erfc(x); }
};

// ---------------------------------------------------------------- Phi, log Phi
struct PhiOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return own::Phi(x); }
};
struct PhiPlatform {  // erfc(-x/sqrt 2)/2
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return 0.5 * Kokkos::erfc(-x * sqrt_half); }
};
struct PhiBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    return 0.5 * boost::math::erfc(-x * sqrt_half, boost_policy());
  }
};
struct LogPhiOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return own::log_Phi(x); }
};
struct LogPhiPlatform {  // log of the naive Phi; log1p of the naive upper tail for x > 0
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    if (x > 0) return Kokkos::log1p(-0.5 * Kokkos::erfc(x * sqrt_half));
    return Kokkos::log(0.5 * Kokkos::erfc(-x * sqrt_half));
  }
};

// ---------------------------------------------------------------- derivatives of log Phi
struct LogPhiDOwn {
  int k;
  double t_switch, c2;
  int m0;
  KOKKOS_INLINE_FUNCTION double operator()(double t) const {
    return own::detail_normal::log_Phi_dk(k, t, t_switch, c2, m0);
  }
};

// statmodels7's pt_logphi.h and pt_enet_logz.h (same author), with R from
// own::inv_mills in place of R's dnorm and pnorm: the recursion for t >= -1,
// below that the cumulants from raw moments, the ratios by the downward
// recursion r_j = j/(x + r_(j+1)) from r_(K+1) = 0 with K = 2000, 300 or 80.
struct LogPhiDSm7 {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double t) const {
    if (t >= -1.0) return own::detail_normal::log_Phi_dk(k, t, -1.0, 0.0, 0);
    const double x = -t;
    const int K = x < 2.0 ? 2000 : (x < 10.0 ? 300 : 80);
    double r = 0.0, rr[6] = {0, 0, 0, 0, 0, 0};
    for (int j = K; j >= 1; --j) {
      r = j / (x + r);
      if (j <= k) rr[j] = r;
    }
    double m[6], kap[6];
    m[0] = 1.0;
    for (int j = 1; j <= k; ++j) m[j] = m[j - 1] * rr[j];
    for (int n = 1; n <= k; ++n) {
      double s = m[n], binom = 1.0;
      for (int i = 1; i < n; ++i) {
        s -= binom * kap[i] * m[n - i];
        binom = binom * (n - i) / i;
      }
      kap[n] = s;
    }
    if (k == 1) return x + kap[1];
    if (k == 2) return -1.0 + kap[2];
    return kap[k];
  }
};

// ---------------------------------------------------------------- quantile
struct PhiInvOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double p) const { return own::Phi_inv(p); }
};
struct PhiInvBoost {  // -sqrt 2 erfc^-1(2p), through 1 - p for p >= 1/2
  KOKKOS_INLINE_FUNCTION double operator()(double p) const {
    if (p < 0.5) return -sqrt2 * boost::math::erfc_inv(2.0 * p, boost_policy());
    return sqrt2 * boost::math::erfc_inv(2.0 * (1.0 - p), boost_policy());
  }
};
struct PhiInvLogpOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double l) const { return own::Phi_inv_logp(l); }
};
struct PhiInvLogpBoost {  // the quantile of exp(l): fails where exp(l) underflows
  KOKKOS_INLINE_FUNCTION double operator()(double l) const { return PhiInvBoost{}(Kokkos::exp(l)); }
};

#ifdef KOKKOS_ENABLE_CUDA
// CUDA's own functions (device only; on the host these return NaN)
struct ErfcxCuda {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    KOKKOS_IF_ON_DEVICE((return ::erfcx(x);))
    KOKKOS_IF_ON_HOST((return x * NAN;))
  }
};
struct PhiCuda {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    KOKKOS_IF_ON_DEVICE((return ::normcdf(x);))
    KOKKOS_IF_ON_HOST((return x * NAN;))
  }
};
struct LogPhiCuda {  // log normcdf, log1p of the upper tail for x > 0
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    KOKKOS_IF_ON_DEVICE((return x > 0 ? ::log1p(-::normcdf(-x)) : ::log(::normcdf(x));))
    KOKKOS_IF_ON_HOST((return x * NAN;))
  }
};
struct PhiInvCuda {
  KOKKOS_INLINE_FUNCTION double operator()(double p) const {
    KOKKOS_IF_ON_DEVICE((return ::normcdfinv(p);))
    KOKKOS_IF_ON_HOST((return p * NAN;))
  }
};
#endif

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_normal <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 20;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };

  const auto t_erfcx = bench::make_time_args(n, [&] { return -3.0 + 30.0 * u01(gen); });
  const auto t_phi = bench::make_time_args(n, [&] { return -12.0 + 20.0 * u01(gen); });
  const auto t_deriv = bench::make_time_args(n, [&] { return -10.0 + 15.0 * u01(gen); });
  const auto t_p = bench::make_time_args(n, [&] { return u01(gen); });
  const auto t_logp = bench::make_time_args(n, [&] { return -expo(-6, 12); });

  auto run = [&](const char* name, const char* impl, auto f, const bench::TimeArgs& t) {
    if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
    bench::report(name, impl, bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t);
  };

  run("erfc", "own", ErfcOwn{}, t_erfcx);
  run("erfc", "platform", ErfcPlatform{}, t_erfcx);
  run("erfc", "boost", ErfcBoost{}, t_erfcx);
  run("erfcx", "own", ErfcxOwn{}, t_erfcx);
  run("erfcx", "naive", ErfcxNaive{}, t_erfcx);
  run("Phi", "own", PhiOwn{}, t_phi);
  run("Phi", "platform", PhiPlatform{}, t_phi);
  run("Phi", "boost", PhiBoost{}, t_phi);
  run("log_Phi", "own", LogPhiOwn{}, t_phi);
  run("log_Phi", "platform", LogPhiPlatform{}, t_phi);
  run("qnorm", "own", PhiInvOwn{}, t_p);
  run("qnorm", "boost", PhiInvBoost{}, t_p);
  run("Phi_inv_logp", "own", PhiInvLogpOwn{}, t_logp);
  run("Phi_inv_logp", "boost", PhiInvLogpBoost{}, t_logp);
#ifdef KOKKOS_ENABLE_CUDA
  run("erfcx", "cuda", ErfcxCuda{}, t_erfcx);
  run("Phi", "cuda", PhiCuda{}, t_phi);
  run("log_Phi", "cuda", LogPhiCuda{}, t_phi);
  run("qnorm", "cuda", PhiInvCuda{}, t_p);
#endif

  // derivatives: the recursion everywhere, statmodels7's port, and the own
  // version over a grid of switch points and lengths of the downward recursion
  static const char* dname[] = {"", "log_Phi_d1", "log_Phi_d2", "log_Phi_d3", "log_Phi_d4", "log_Phi_d5"};
  // the recursion everywhere (to show its cancellation), statmodels7's port,
  // and the own version (recursion from t = -1, K = k + 20 + ceil(300 / t^2))
  for (int k = 1; k <= 5; ++k) {
    run(dname[k], "recursion", LogPhiDOwn{k, -1e300, 0.0, 0}, t_deriv);
    run(dname[k], "sm7", LogPhiDSm7{k}, t_deriv);
    run(dname[k], "own", LogPhiDOwn{k, -1.0, 300.0, 20}, t_deriv);
  }
  return 0;
}
