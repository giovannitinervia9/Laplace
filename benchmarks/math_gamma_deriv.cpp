// Accuracy and speed of the derivatives of log P(a, x) and log Q(a, x) with
// respect to a, orders 1 to 5 (include/laplace/math/incomplete_gamma_deriv.hpp).
// Candidates: the chosen route (own: the series on the lower side where it is
// short, the quadrature elsewhere); the double-exponential quadrature in
// d = log(t/x) or u = t - x with 8 nodes per unit of t (quad8) and with 6 and
// 12 (quad6, quad12); the series and continued fraction differentiated term by
// term (series); the series on every lower side with the quadrature on the
// upper side (ser+quad); and, as a baseline, a central finite difference of
// own::gamma_inc's log P or log Q in a (fd).
//
// Each line of bench::report is followed by a breakdown by region: a < 1;
// a > 1000; for 1 <= a <= 1000 the far lower tail (log P < -20), the far upper
// tail (log Q < -20) and the rest (mid). ulpL = ulp / max(1, |log|f||).
//
// Usage: math_gamma_deriv <reference dir> <output dir> [name filter] [impl filter]
#include "measure.hpp"

#include <laplace/math/incomplete_gamma_deriv.hpp>

#include <random>

namespace {

namespace gd = laplace::math::own::detail_gamma_deriv;

struct Deriv {
  int k;
  bool p;
  int method;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    return gd::log_gamma_da(k, a, x, p, method);
  }
};

// Central differences of order two in the step, step h = a c eps^(1/(k+2)),
// capped at a/4 so that a - 3h > 0.
struct FiniteDiff {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double f(double a, double x) const {
    const auto r = laplace::math::own::gamma_inc(a, x);
    return p ? r.log_p : r.log_q;
  }
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    const double e = Kokkos::pow(2.220446049250313e-16, 1.0 / (k + 2));
    double h = a * e;
    if (h > 0.25 * a) h = 0.25 * a;
    switch (k) {
      case 1: return (f(a + h, x) - f(a - h, x)) / (2 * h);
      case 2: return (f(a + h, x) - 2 * f(a, x) + f(a - h, x)) / (h * h);
      case 3: return (f(a + 2 * h, x) - 2 * f(a + h, x) + 2 * f(a - h, x) - f(a - 2 * h, x)) / (2 * h * h * h);
      case 4:
        return (f(a + 2 * h, x) - 4 * f(a + h, x) + 6 * f(a, x) - 4 * f(a - h, x) + f(a - 2 * h, x)) /
               (h * h * h * h);
      default:
        return (f(a + 3 * h, x) - 4 * f(a + 2 * h, x) + 5 * f(a + h, x) - 5 * f(a - h, x) + 4 * f(a - 2 * h, x) -
                f(a - 3 * h, x)) /
               (2 * h * h * h * h * h);
    }
  }
};

const char* region_name[5] = {"a<1", "a>1000", "lowtail", "uptail", "mid"};

int region_of(double a, double x) {
  if (a < 1.0) return 0;
  if (a > 1000.0) return 1;
  const auto r = laplace::math::own::gamma_inc(a, x);
  if (r.log_p < -20.0) return 2;
  if (r.log_q < -20.0) return 3;
  return 4;
}

void breakdown(const bench::Ref& r, const std::vector<double>& y) {
  double ulp[5] = {0}, ulp1[5] = {0}, ulpL[5] = {0};
  bench::Args worst[5]{}, worstL[5]{};
  int cnt[5] = {0}, bad[5] = {0};
  for (std::size_t i = 0; i < y.size(); ++i) {
    const int g = region_of(r.x[i][0], r.x[i][1]);
    ++cnt[g];
    const double hi = r.hi[i];
    const double e = std::fabs(bench::error_in(y[i], hi, r.lo[i], std::max(std::fabs(hi), 2.2250738585072014e-308)));
    if (std::isinf(e)) {
      ++bad[g];
      continue;
    }
    const double e1 = std::fabs(bench::error_in(y[i], hi, r.lo[i], std::max(std::fabs(hi), 1.0)));
    const double eL = e / std::max(1.0, std::fabs(std::log(std::fabs(hi))));
    if (e > ulp[g]) ulp[g] = e, worst[g] = r.x[i];
    ulp1[g] = std::max(ulp1[g], e1);
    if (eL > ulpL[g]) ulpL[g] = eL, worstL[g] = r.x[i];
  }
  for (int g = 0; g < 5; ++g) {
    std::printf("    %-8s n %4d: max %9.4g ulp, %8.4g ulp1, %8.4g ulpL, worst (%s)%s\n", region_name[g], cnt[g], ulp[g],
                ulp1[g], ulpL[g], bench::show(worst[g], 2).c_str(), bad[g] ? " NON-FINITE" : "");
  }
}


// ---------------------------------------------------------------- profile: where the time goes

// the parts of one call, each as a functor of (a, x)
struct PQuad {  // quadrature moments on the side the route takes for P
  int e;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    double kap[6], li;
    const bool lower = gd::side_lower(a, x, true);
    if (e == 1) gd::quad_moments<8, 1>(a, x, lower, kap, li);
    else gd::quad_moments<8, 0>(a, x, lower, kap, li);
    return kap[1] + kap[5] + li;
  }
};
struct PSeries {  // series moments on that side (nmax huge; upper only for x >= 36)
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    double kap[6], li;
    const bool lower = gd::side_lower(a, x, true);
    if (!lower && x < 36.0) return 0.0;
    if (!gd::series_moments(a, x, lower, 5, 100000, kap, li)) return -1.0;
    return kap[1] + kap[5] + li;
  }
};
struct PNodes {  // the node map alone: 60 nodes of the exp-exp rule
  int computed;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    double s = 0;
    for (int j = -29; j <= 30; ++j) {
      double xi, wq;
      if (computed) gd::ee_node<6, 0>(j, a < x, xi, wq);  // NH = 6 has no table: computed
      else gd::ee_node<8, 0>(j, a < x, xi, wq);
      s += xi * wq;
    }
    return s;
  }
};
struct PExp {  // 60 integrand evaluations without the moments: exp and expm1mx
  int e;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    double s = 0, d = -0.01 * a / (x + 1.0);
    for (int j = 0; j < 60; ++j) {
      const double v = (a - x) * d - x * gd::expm1mx(d);
      s += e ? laplace::math::own::exp(v) : laplace::math::exp(v);
      d *= 1.05;
    }
    return s;
  }
};
struct PMoments {  // 60 updates of the central moments
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    gd::central c;
    double d = a, w = x;
    for (int j = 0; j < 60; ++j) {
      c.add(w, d);
      d *= 1.01;
      w *= 0.97;
    }
    return c.M5 + c.m;
  }
};
struct PPoly {  // psi^(k-1) for k = 2..5
  KOKKOS_INLINE_FUNCTION double operator()(double a, double) const {
    double s = 0;
    for (int k = 1; k <= 4; ++k) s += laplace::math::polygamma(k, a);
    return s;
  }
};
struct PPsiLog {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double) const { return gd::psi_minus_log(a); }
};
struct PPrefactor {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    return laplace::math::own::detail_igamma::log_prefactor(a, x);
  }
};
struct PChoose {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return gd::side_lower(a, x, true) ? 1.0 : 0.0; }
};
struct PGammaInc {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    return laplace::math::own::gamma_inc(a, x).log_p;
  }
};

template <class F>
void ptime(const char* region, const char* part, const bench::TimeArgs& t, F f) {
  const double s1 = bench::time_ns<bench::HostOne>(t.host, f, 3);
  const double sx = bench::time_ns<bench::Exec>(t.dev, f, 5);
  std::printf("%-10s %-22s ns/call Serial %9.2f, %s %8.3f\n", region, part, s1, bench::Exec::name(), sx);
  std::fflush(stdout);
}

void profile(std::size_t n, std::mt19937_64& gen) {
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> z(0.0, 1.0);
  auto exp2u = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };
  struct R {
    const char* name;
    bench::TimeArgs t;
  };
  std::vector<R> regions;
  regions.push_back({"all", bench::make_time_args(n, [&] {
                       const double a = exp2u(-3, 10);
                       return bench::Args{a, a * std::exp(0.5 * z(gen)), 0};
                     })});
  regions.push_back({"a<1", bench::make_time_args(n, [&] {
                       const double a = exp2u(-10, 0);
                       return bench::Args{a, a * std::exp(z(gen)), 0};
                     })});
  regions.push_back({"centre1-50", bench::make_time_args(n, [&] {
                       const double a = 1.0 + 49.0 * u01(gen);
                       return bench::Args{a, std::max(1e-3, a + z(gen) * std::sqrt(a)), 0};
                     })});
  regions.push_back({"centre>50", bench::make_time_args(n, [&] {
                       const double a = exp2u(5.6, 13.3);
                       return bench::Args{a, a + z(gen) * std::sqrt(a), 0};
                     })});
  regions.push_back({"lowtail", bench::make_time_args(n, [&] {
                       const double a = exp2u(0, 13);
                       return bench::Args{a, a * (0.01 + 0.59 * u01(gen)), 0};
                     })});
  regions.push_back({"uptail", bench::make_time_args(n, [&] {
                       const double a = exp2u(-3, 13);
                       return bench::Args{a, a + std::sqrt(a) * (6.0 + 24.0 * u01(gen)) + 5.0, 0};
                     })});
  for (const auto& r : regions) {
    ptime(r.name, "P k1 own", r.t, Deriv{1, true, 0});
    ptime(r.name, "P k5 own", r.t, Deriv{5, true, 0});
    ptime(r.name, "Q k1 own", r.t, Deriv{1, false, 0});
    ptime(r.name, "Q k5 own", r.t, Deriv{5, false, 0});
    ptime(r.name, "P k5 quad8", r.t, Deriv{5, true, 5});
    ptime(r.name, "P k5 no Temme", r.t, Deriv{5, true, 31});
    ptime(r.name, "Q k5 no Temme", r.t, Deriv{5, false, 31});
    ptime(r.name, "P k5 oe-own", r.t, Deriv{5, true, 6});
    ptime(r.name, "P k5 oe-quad8", r.t, Deriv{5, true, 7});
    ptime(r.name, "P k5 first round", r.t, Deriv{5, true, 4});
    ptime(r.name, "quad moments", r.t, PQuad{0});
    ptime(r.name, "quad moments own exp", r.t, PQuad{1});
    ptime(r.name, "series moments", r.t, PSeries{});
    ptime(r.name, "60 nodes, table", r.t, PNodes{0});
    ptime(r.name, "60 nodes, computed", r.t, PNodes{1});
    ptime(r.name, "60 integrands", r.t, PExp{0});
    ptime(r.name, "60 integrands own exp", r.t, PExp{1});
    ptime(r.name, "60 moment updates", r.t, PMoments{});
    ptime(r.name, "4 polygammas", r.t, PPoly{});
    ptime(r.name, "psi - log", r.t, PPsiLog{});
    ptime(r.name, "log prefactor", r.t, PPrefactor{});
    ptime(r.name, "side choice", r.t, PChoose{});
    ptime(r.name, "gamma_inc", r.t, PGammaInc{});
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_gamma_deriv <reference dir> <output dir> [name filter] [impl filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  const std::string ifilter = argc > 4 ? argv[4] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 15;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> z(0.0, 1.0);
  const auto t_gamma = bench::make_time_args(n, [&] {
    const double a = std::exp2(-3.0 + 13.0 * u01(gen));
    return bench::Args{a, a * std::exp(0.5 * z(gen)), 0};
  });

  if (ifilter == "profile") {
    profile(n, gen);
    return 0;
  }

  auto run = [&](const std::string& name, const char* impl, auto f) {
    if (!filter.empty() && name.find(filter) == std::string::npos) return;
    if (!ifilter.empty() && std::string(impl).find(ifilter) == std::string::npos) return;
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    bench::report(name.c_str(), impl, r, f, outdir, t_gamma);
    breakdown(r, bench::eval_kernel(r.x, f));
    std::fflush(stdout);
  };

  for (int k = 1; k <= 5; ++k) {
    for (int side = 0; side < 2; ++side) {
      const bool p = side == 0;
      const std::string name = std::string(p ? "log_gamma_p_da" : "log_gamma_q_da") + std::to_string(k);
      run(name, "own", Deriv{k, p, 0});
      run(name, "quad8", Deriv{k, p, 5});
      run(name, "noTemme", Deriv{k, p, 31});
      run(name, "quad6", Deriv{k, p, 2});
      run(name, "quad12", Deriv{k, p, 3});
      run(name, "series", Deriv{k, p, 1});
      run(name, "ser+quad", Deriv{k, p, 4});
      run(name, "fd", FiniteDiff{k, p});
      run(name, "oe-own", Deriv{k, p, 6});
      run(name, "oe-quad8", Deriv{k, p, 7});
      run(name, "nmax40", Deriv{k, p, 10});
      run(name, "nmax80", Deriv{k, p, 11});
      run(name, "nmax160", Deriv{k, p, 12});
      run(name, "nmax240", Deriv{k, p, 13});
    }
  }
  return 0;
}
