// Accuracy and speed of Owen's T and of the log distribution function of the
// standard skew normal with its derivatives in (z, alpha) to order five
// (include/laplace/math/owen.hpp). Candidates:
//   owens_t     own      the chosen algorithm (Gauss rules, Owen's identity, Jbar);
//               direct   16-point Gauss-Legendre on the defining integral for
//                        a <= 1 and Owen's identity in its usual form
//                        T(h, a) = (Q(h) + Q(ah))/2 - Q(h) Q(ah) - T(ah, 1/a) for a > 1
//                        (a baseline, on the device);
//               boost    boost::math::owens_t (Patefield-Tandy; host only, gcc build).
//   log_sn_cdf  own      the chosen algorithm;
//               naive    log(Phi(z) - 2 T(z, alpha)) with the own T (device);
//               boost    the same with Boost's T (host only).
//   derivatives own      Jbar's moments in the lower tails, the elementary
//                        series elsewhere;
//               naive    the elementary series of F everywhere, with the own F.
// Errors per component as in benchmarks/math_beta_deriv.cpp: ulp (relative, floored
// at 2^-1022), ulp1 (in ulps of max(|f|, 1)), ulpN (in ulps of max(|f|, N),
// N = |f_K0|^(i/K) |f_0K|^(j/K) the geometric mean of the pure components of
// the order), ulpL = ulpN / max(1, |log |f||).
// Regions of (z, alpha): centre |z| <= 2; lower+ z < -2, alpha > 0; lower-
// z < -2, alpha < 0; upper z > 2; small |alpha| < 1/8; large |alpha| > 8 (the
// last two overlap the others). For owens_t, with h, a: centre |h| <= 2;
// tail |h| > 2 and |a| <= 1; tail-a |h| > 2 and |a| > 1; small |a| < 1/8;
// large |a| > 8.
//
// Constants of the algorithm through the environment: LAPLACE_OW_HN, _BA, _BT,
// LAPLACE_OW_BL, LAPLACE_OW_NH, LAPLACE_OW_NL, LAPLACE_OW_BD, LAPLACE_OW_DH, LAPLACE_OW_CR,
// LAPLACE_OW_CE, LAPLACE_OW_TC, and the same with suffix 0 for the value alone (detail_owen::params).
//
// Usage: math_owen <reference dir> [acc|all|worst|time] [filter]
#include "measure.hpp"

#include <laplace/math/owen.hpp>

#include <boost/math/special_functions/owens_t.hpp>

#include <random>

namespace {

namespace own = laplace::math::own;
namespace dow = laplace::math::own::detail_owen;
using laplace::math::detail::boost_policy;

dow::params env_params() {
  dow::params p;
  if (const char* s = std::getenv("LAPLACE_OW_HN")) p.h_n = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_BA")) p.beta_a = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_BT")) p.beta_t = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_DH")) p.d_h = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_CR")) p.gl_cr = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_CE")) p.gl_ce = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_TC")) p.t_cut = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_BL")) p.beta_l = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_NH")) p.n_h = std::atoi(s);
  if (const char* s = std::getenv("LAPLACE_OW_NL")) p.n_l = std::atoi(s);
  if (const char* s = std::getenv("LAPLACE_OW_BD")) p.beta_d = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_DH0")) p.d_h0 = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_BL0")) p.beta_l0 = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_CR0")) p.gl_cr0 = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_CE0")) p.gl_ce0 = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_TC0")) p.t_cut0 = std::atof(s);
  if (const char* s = std::getenv("LAPLACE_OW_NH0")) p.n_h0 = std::atoi(s);
  if (const char* s = std::getenv("LAPLACE_OW_NL0")) p.n_l0 = std::atoi(s);
  return p;
}

// ---------------------------------------------------------------- candidates
struct OwenOwn {
  dow::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double h, double a) const { return dow::owens_t(h, a, p); }
};
#ifndef KOKKOS_ENABLE_CUDA
struct OwenBoost {
  double operator()(double h, double a) const { return boost::math::owens_t(h, a, boost_policy()); }
};
#endif
// T for h >= 0, a >= 0 by the defining integral (a <= 1) and the usual identity (a > 1)
KOKKOS_INLINE_FUNCTION double owen_direct_pos(double h, double a) {
  if (a <= 1.0) return dow::inv_2pi * laplace::math::own::detail_normal::exp_minus_half_square(h) * dow::t_direct(h, a);
  const double b = 1.0 / a, ah = a * h;
  const double qh = own::Phi(-h), qa = own::Phi(-ah);
  const double t2 = dow::inv_2pi * laplace::math::own::detail_normal::exp_minus_half_square(ah) * dow::t_direct(ah, b);
  return 0.5 * (qh + qa) - qh * qa - t2;
}
struct OwenDirect {
  KOKKOS_INLINE_FUNCTION double operator()(double h, double a) const {
    const double s = a < 0 ? -1.0 : 1.0;
    return s * owen_direct_pos(Kokkos::fabs(h), Kokkos::fabs(a));
  }
};
struct SnOwn {
  dow::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double z, double alpha) const { return dow::log_sn_cdf(z, alpha, p); }
};
struct SnNaive {
  dow::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double z, double alpha) const {
    return Kokkos::log(own::Phi(z) - 2.0 * dow::owens_t(z, alpha, p));
  }
};
#ifndef KOKKOS_ENABLE_CUDA
struct SnBoost {
  double operator()(double z, double alpha) const {
    return std::log(own::Phi(z) - 2.0 * boost::math::owens_t(z, alpha, boost_policy()));
  }
};
#endif

template <int K>
struct Deriv {
  dow::params p;
  int method;
  KOKKOS_INLINE_FUNCTION void eval(double z, double alpha, double* o) const {
    const auto r = dow::sn_deriv<K>(z, alpha, p, method);
    for (int j = 0; j <= K; ++j) o[j] = r.c[j];
  }
};
template <int K>
struct Sum {
  Deriv<K> d;
  KOKKOS_INLINE_FUNCTION double operator()(double z, double alpha) const {
    double o[K + 1];
    d.eval(z, alpha, o);
    double s = 0;
    for (int j = 0; j <= K; ++j) s += o[j];
    return s;
  }
};

// ---------------------------------------------------------------- regions
constexpr int n_reg = 6;
const char* sn_region_name[n_reg] = {"centre", "lower+", "lower-", "upper", "small-a", "large-a"};
const char* ow_region_name[n_reg] = {"centre", "tail", "tail-a", "-", "small-a", "large-a"};
// the regions a row belongs to (bit k for region k)
int sn_regions(double z, double alpha) {
  int m = 0;
  if (std::fabs(z) <= 2) m |= 1;
  else if (z < -2 && alpha > 0) m |= 2;
  else if (z < -2) m |= 4;
  else m |= 8;
  if (std::fabs(alpha) < 0.125) m |= 16;
  if (std::fabs(alpha) > 8) m |= 32;
  return m;
}
int ow_regions(double h, double a) {
  int m = 0;
  if (std::fabs(h) <= 2) m |= 1;
  else if (std::fabs(a) <= 1) m |= 2;
  else m |= 4;
  if (std::fabs(a) < 0.125) m |= 16;
  if (std::fabs(a) > 8) m |= 32;
  return m;
}

constexpr double tiny = 2.2250738585072014e-308;

// value functions: errors with the region breakdown
template <class F>
void value_report(const char* fn, const char* impl, const bench::Ref& r, F f, const bench::TimeArgs& t, bool device,
                  bool owen, bool worst, bool timing) {
  const auto yh = bench::eval_host(r.x, f);
  const auto y = device ? bench::eval_kernel(r.x, f) : yh;
  double reg[n_reg] = {0, 0, 0, 0, 0, 0};
  double mu = 0, mu1 = 0, muL = 0, mhost = 0;
  std::size_t bad = 0, wi = 0;
  struct Row {
    double e;
    std::size_t i;
  };
  std::vector<Row> rows;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i]) && std::isfinite(r.hi[i])) {
      if (bad++ == 0) std::printf("  first non-finite at %s: %g\n", bench::show(r.x[i], 2).c_str(), y[i]);
      continue;
    }
    const double d = std::fabs((y[i] - r.hi[i]) - r.lo[i]);
    const double u = d / bench::ulp_of(std::max(std::fabs(r.hi[i]), tiny));
    const double u1 = d / bench::ulp_of(std::max(std::fabs(r.hi[i]), 1.0));
    const double uL = r.hi[i] != 0 ? u / std::max(1.0, std::fabs(std::log(std::fabs(r.hi[i])))) : u;
    const double uh = std::fabs((yh[i] - r.hi[i]) - r.lo[i]) / bench::ulp_of(std::max(std::fabs(r.hi[i]), tiny));
    if (u > mu) mu = u, wi = i;
    mu1 = std::max(mu1, u1);
    muL = std::max(muL, uL);
    mhost = std::max(mhost, uh);
    const int m = owen ? ow_regions(r.x[i][0], r.x[i][1]) : sn_regions(r.x[i][0], r.x[i][1]);
    for (int k = 0; k < n_reg; ++k)
      if (m & (1 << k)) reg[k] = std::max(reg[k], u);
    if (worst) rows.push_back({u, i});
  }
  char tm[96] = "";
  if (timing) {
    const double t1 = bench::time_ns<bench::HostOne>(t.host, f, 5);
    if (device)
      std::snprintf(tm, sizeof tm, " | ns/call Serial %.1f, %s %.3f", t1, bench::Exec::name(),
                    bench::time_ns<bench::Exec>(t.dev, f, 7));
    else
      std::snprintf(tm, sizeof tm, " | ns/call Serial %.1f", t1);
  }
  std::printf("%-10s %-7s | %s: max %9.3g ulp (at %s), %8.3g ulp1, %8.3g ulpL | host max %9.3g ulp | bad %zu%s\n", fn, impl,
              device ? bench::Exec::name() : "host", mu, bench::show(r.x[wi], 2).c_str(), mu1, muL, mhost, bad, tm);
  std::printf("           ulp by region:");
  const char** names = owen ? ow_region_name : sn_region_name;
  for (int k = 0; k < n_reg; ++k)
    if (names[k][0] != '-') std::printf(" %s %.3g", names[k], reg[k]);
  std::printf("\n");
  if (worst) {
    std::sort(rows.begin(), rows.end(), [](const Row& p, const Row& q) { return p.e > q.e; });
    for (std::size_t k = 0; k < std::min<std::size_t>(rows.size(), 8); ++k)
      std::printf("      %9.3g ulp at %-26s ref %.17g got %.17g\n", rows[k].e, bench::show(r.x[rows[k].i], 2).c_str(),
                  r.hi[rows[k].i], y[rows[k].i]);
  }
}

// ---------------------------------------------------------------- derivatives
struct Refs {
  bench::Ref r[6][6];  // r[i][j], 1 <= i + j <= 5
};

Refs load(const std::string& dir) {
  Refs R;
  for (int n = 1; n <= 5; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      const std::string name = "log_sn_cdf_d" + std::to_string(i) + std::to_string(j);
      R.r[i][j] = bench::read_ref(dir + "/" + name + ".csv");
      if (R.r[i][j].x != R.r[1][0].x) {
        std::fprintf(stderr, "%s: arguments differ\n", name.c_str());
        std::exit(2);
      }
    }
  return R;
}

template <int K>
std::vector<std::array<double, 6>> eval_rows(const std::vector<bench::Args>& x, Deriv<K> c) {
  const std::size_t n = x.size();
  const auto hx = bench::to_host_args(x);
  bench::DArgs dx("x", n);
  Kokkos::deep_copy(dx, hx);
  Kokkos::View<double* [6], Kokkos::LayoutRight, bench::Exec::memory_space> dy("y", n);
  Kokkos::parallel_for(
      "eval", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t r) {
        double o[K + 1];
        c.eval(dx(r, 0), dx(r, 1), o);
        for (int j = 0; j <= K; ++j) dy(r, j) = o[j];
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<std::array<double, 6>> y(n);
  for (std::size_t r = 0; r < n; ++r)
    for (int j = 0; j <= K; ++j) y[r][j] = hy(r, j);
  return y;
}

template <int K>
void deriv_report(const Refs& R, const char* name, Deriv<K> c, const bench::TimeArgs& t, bool worst, bool timing) {
  const auto y = eval_rows<K>(R.r[1][0].x, c);
  const auto& x = R.r[1][0].x;
  double eu = 0, e1 = 0, eN = 0, eL = 0, reg[n_reg] = {0, 0, 0, 0, 0, 0};
  std::size_t bad = 0, wr = 0;
  int wj = 0;
  struct Row {
    double eN, eU;
    std::size_t r;
    int j;
  };
  std::vector<Row> rows;
  for (std::size_t r = 0; r < x.size(); ++r) {
    const double fK0 = R.r[K][0].hi[r], f0K = R.r[0][K].hi[r];
    for (int j = 0; j <= K; ++j) {
      const int i = K - j;
      const auto& ref = R.r[i][j];
      const double hi = ref.hi[r], lo = ref.lo[r], v = y[r][j];
      if (!std::isfinite(v)) {
        if (bad++ == 0) std::printf("  first non-finite d%d%d at %s\n", i, j, bench::show(x[r], 2).c_str());
        continue;
      }
      const double N = std::pow(std::fabs(fK0), double(i) / K) * std::pow(std::fabs(f0K), double(j) / K);
      const double d = std::fabs((v - hi) - lo);
      const double u = d / bench::ulp_of(std::max(std::fabs(hi), tiny));
      const double u1 = d / bench::ulp_of(std::max(std::fabs(hi), 1.0));
      const double uN = d / bench::ulp_of(std::max({std::fabs(hi), N, tiny}));
      if (u > eu) eu = u, wr = r, wj = j;
      e1 = std::max(e1, u1);
      eN = std::max(eN, uN);
      if (hi != 0.0) eL = std::max(eL, uN / std::max(1.0, std::fabs(std::log(std::fabs(hi)))));
      const int m = sn_regions(x[r][0], x[r][1]);
      for (int k = 0; k < n_reg; ++k)
        if (m & (1 << k)) reg[k] = std::max(reg[k], uN);
      if (worst) rows.push_back({uN, u, r, j});
    }
  }
  char tm[96] = "";
  if (timing) {
    const Sum<K> f{c};
    std::snprintf(tm, sizeof tm, " | ns/call Serial %.1f, %s %.3f", bench::time_ns<bench::HostOne>(t.host, f, 3),
                  bench::Exec::name(), bench::time_ns<bench::Exec>(t.dev, f, 5));
  }
  std::printf("%-6s K=%d | max %9.3g ulp (d%d%d at %s), %8.3g ulp1, %8.3g ulpN, %8.3g ulpL | bad %zu%s\n", name, K, eu,
              K - wj, wj, bench::show(x[wr], 2).c_str(), e1, eN, eL, bad, tm);
  std::printf("       ulpN by region:");
  for (int k = 0; k < n_reg; ++k) std::printf(" %s %.3g", sn_region_name[k], reg[k]);
  std::printf("\n");
  if (worst) {
    std::sort(rows.begin(), rows.end(), [](const Row& p, const Row& q) { return p.eN > q.eN; });
    for (std::size_t k = 0; k < std::min<std::size_t>(rows.size(), 6); ++k) {
      const auto& w = rows[k];
      std::printf("      ulpN %9.3g ulp %9.3g d%d%d  z %-12.6g alpha %-12.6g ref %.10g got %.10g\n", w.eN, w.eU, K - w.j,
                  w.j, x[w.r][0], x[w.r][1], R.r[K - w.j][w.j].hi[w.r], y[w.r][w.j]);
    }
  }
}

template <int K>
void run_order(const Refs& R, const std::string& filter, const bench::TimeArgs& t, bool worst, bool timing,
               const dow::params& p) {
  if (filter.empty() || filter == "own") deriv_report<K>(R, "own", Deriv<K>{p, 0}, t, worst, timing);
  if (filter.empty() || filter == "naive") deriv_report<K>(R, "naive", Deriv<K>{p, 1}, t, worst, timing);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_owen <reference dir> [acc|all|worst|time] [filter]\n");
    return 2;
  }
  const std::string refdir = argv[1];
  const std::string mode = argc > 2 ? argv[2] : "acc";
  const std::string filter = argc > 3 ? argv[3] : "";
  const bool worst = mode == "worst", timing = mode == "all" || mode == "time";
  bench::print_spaces();
  const dow::params p = env_params();
  if (mode == "point") {  // math_owen - point z alpha: internals on the host
    const double z = std::atof(argv[3]), al = std::atof(argv[4]);
    const double h = std::fabs(z), a = std::fabs(al);
    std::printf("log_E(|z|, |alpha|) %.17g  owens_t %.17g  log_sn_cdf %.17g\n", dow::lower(h, a, p).l, dow::owens_t(z, al, p),
                dow::log_sn_cdf(z, al, p));
    double L[dow::nc<5>];
    dow::log_jbar<5>(h, a, p, L);
    std::printf("log Jbar series:");
    for (int k = 0; k < dow::nc<5>; ++k) std::printf(" %.17g", L[k]);
    std::printf("\n");
    for (int m = 0; m < 2; ++m) {
      double S[dow::nc<5>];
      dow::scale sp, sm;
      dow::sn_series<5>(z, al, p, m, S, sp, sm);
      std::printf("sn_series method %d (scales %.17g %.17g, %.17g %.17g):", m, sp.S, sp.m, sm.S, sm.m);
      for (int k = 0; k < dow::nc<5>; ++k) std::printf(" %.17g", S[k]);
      std::printf("\n");
    }
    return 0;
  }
  std::printf("params: h_n %g, beta_a %g, beta_t %g, d_h %g, beta_l %g, n_h %d, n_l %d, gl_cr %g, gl_ce %g, t_cut %g, beta_d %g\n", p.h_n, p.beta_a, p.beta_t, p.d_h,
              p.beta_l, p.n_h, p.n_l, p.gl_cr, p.gl_ce, p.t_cut, p.beta_d);

  // timing arguments: alpha (and a) = +-2^U(-4, 5), z (and h) = 2 Z
  const std::size_t n = std::size_t(1) << 16;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> nz(0.0, 1.0);
  const auto t = bench::make_time_args(n, [&] {
    const double al = std::exp2(-4.0 + 9.0 * u01(gen)) * (u01(gen) < 0.5 ? -1.0 : 1.0);
    return bench::Args{2.0 * nz(gen), al, 0.0};
  });

  if (mode == "prof") {  // ns/call of the pieces on the timing arguments (h = |z|, a = |alpha|)
    auto tm = [&](const char* what, auto f) {
      std::printf("  %-34s Serial %8.1f  %s %8.3f\n", what, bench::time_ns<bench::HostOne>(t.host, f, 5), bench::Exec::name(),
                  bench::time_ns<bench::Exec>(t.dev, f, 7));
    };
    tm("Kokkos::exp", KOKKOS_LAMBDA(double z, double) { return Kokkos::exp(-0.5 * z * z); });
    tm("Kokkos::erf", KOKKOS_LAMBDA(double z, double) { return Kokkos::erf(z); });
    tm("own::Phi", KOKKOS_LAMBDA(double z, double) { return own::Phi(z); });
    tm("own::erfcx", KOKKOS_LAMBDA(double z, double) { return own::erfcx(Kokkos::fabs(z)); });
    tm("t_direct(h, min(a, 1))", KOKKOS_LAMBDA(double z, double a) {
      return dow::t_direct(Kokkos::fabs(z), Kokkos::fmin(Kokkos::fabs(a), 1.0));
    });
    tm("log_jbar<0>(h + 1, a)", KOKKOS_LAMBDA(double z, double a) {
      double L[1];
      dow::log_jbar<0>(Kokkos::fabs(z) + 1.0, Kokkos::fabs(a), p, L);
      return L[0];
    });
    tm("log_jbar<5>(h + 1, a)", KOKKOS_LAMBDA(double z, double a) {
      double L[dow::nc<5>];
      dow::log_jbar<5>(Kokkos::fabs(z) + 1.0, Kokkos::fabs(a), p, L);
      return L[20];
    });
    tm("log_g0_series<5>(h + 1)", KOKKOS_LAMBDA(double z, double) {
      double L[dow::nc<5>];
      dow::log_g0_series<5>(Kokkos::fabs(z) + 1.0, L);
      return L[15];
    });
    tm("lower(|z|, alpha)", KOKKOS_LAMBDA(double z, double a) { return dow::lower(Kokkos::fabs(z), a, p).l; });
    tm("owens_t", KOKKOS_LAMBDA(double z, double a) { return dow::owens_t(z, a, p); });
    long cnt[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < t.host.extent(0); ++i) {
      const double z = t.host(i, 0), a = t.host(i, 1);
      if (z <= 0) ++cnt[dow::region(-z, a, p)];
    }
    std::printf("regions of the timing arguments with z <= 0: %ld %ld %ld %ld\n", cnt[0], cnt[1], cnt[2], cnt[3]);
    return 0;
  }
  auto want = [&](const char* s) { return filter.empty() || filter == s; };
  if (want("owens_t") || want("value")) {
    const auto r = bench::read_ref(refdir + "/owens_t.csv");
    value_report("owens_t", "own", r, OwenOwn{p}, t, true, true, worst, timing);
    value_report("owens_t", "direct", r, OwenDirect{}, t, true, true, worst, timing);
#ifndef KOKKOS_ENABLE_CUDA
    value_report("owens_t", "boost", r, OwenBoost{}, t, false, true, worst, timing);
#endif
  }
  if (want("log_sn_cdf") || want("value")) {
    const auto r = bench::read_ref(refdir + "/log_sn_cdf.csv");
    value_report("log_sn_cdf", "own", r, SnOwn{p}, t, true, false, worst, timing);
    value_report("log_sn_cdf", "naive", r, SnNaive{p}, t, true, false, worst, timing);
#ifndef KOKKOS_ENABLE_CUDA
    value_report("log_sn_cdf", "boost", r, SnBoost{}, t, false, false, worst, timing);
#endif
  }
  if (!(want("owens_t") || want("log_sn_cdf") || want("value")) || filter.empty()) {
    const Refs R = load(refdir);
    const std::string df = (filter == "own" || filter == "naive") ? filter : "";
    run_order<1>(R, df, t, worst, timing, p);
    run_order<2>(R, df, t, worst, timing, p);
    run_order<3>(R, df, t, worst, timing, p);
    run_order<4>(R, df, t, worst, timing, p);
    run_order<5>(R, df, t, worst, timing, p);
  }
  return 0;
}
