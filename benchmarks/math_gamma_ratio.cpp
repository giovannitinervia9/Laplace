// Accuracy and speed of the partial derivatives of log P(a, a c) and
// log Q(a, a c) in a (at fixed c) and c, total orders 1 to 6
// (include/laplace/math/incomplete_gamma_ratio.hpp), and of the derivatives
// in x of log P and log Q at integer a near the mean (the Poisson cdf,
// incomplete_gamma_mixed.hpp).
//
// Candidates (ratio files): the chosen route (own); the chosen route with the
// series limits of the device, run on the host (devroute); the quadrature
// alone (quad); the chosen route without Temme (noTemme) and with Temme from
// a = 20 as in incomplete_gamma_mixed.hpp (Temme20); the series in
// (h, e) of incomplete_gamma_mixed.hpp composed with e = h/a + g + h g/a for
// every a (fallback: the composition over (a, x) that the gamma distribution
// used before). Poisson files: the chosen route of incomplete_gamma_mixed.hpp
// (own), the same with the binomial products in its density ratio for every a
// (products, before 2026-11-12, evening), without the density ratio
// (noratio), without Temme (noTemme) and with Temme for every component
// (allTemme, before 2026-11-12, evening).
//
// Figures, besides the standard line of measure.hpp: "S" the error in ulp of
// the largest component of its order at that point in the logarithmic
// variables (ratio files: max_j |a^(K-j) c^j v_j| / (a^(K-j) c^j)) or of the
// largest |value| of the component over the x of the same a (Poisson files),
// by component j and by route ("Temme region": the region of Temme's
// expansion in incomplete_gamma_mixed.hpp, whatever the route taken there);
// "O" the error in ulp of the component itself, where it is above 1e-12 of
// that scale.
//
// Usage: math_gamma_ratio <reference dir> <output dir> [name filter] [impl filter]
#include "measure.hpp"

#include <laplace/math/incomplete_gamma_ratio.hpp>

#include <map>
#include <random>

namespace {

namespace gd = laplace::math::own::detail_gamma_deriv;
namespace gr = laplace::math::own::detail_gamma_ratio;
namespace gm = laplace::math::own::detail_gamma_mixed;

struct RatioF {
  int k;
  bool p;
  int method;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double c, double j) const {
    double v[7];
    gr::log_gamma_ratio(k, a, c, p, method, v);
    return v[static_cast<int>(j)];
  }
};

struct MixedF {
  int k;
  bool p;
  int method;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x, double j) const {
    double v[7];
    gm::log_gamma_mixed(k, a, x, p, method, v);
    return v[static_cast<int>(j)];
  }
};

// the public functions, for the timing of one call
struct PublicRatio {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double c, double) const {
    namespace lm = laplace::math;
    if (p) {
      switch (k) {
        case 1: return lm::log_gamma_p_ratio_mixed_d1(a, c).v[1];
        case 2: return lm::log_gamma_p_ratio_mixed_d2(a, c).v[1];
        case 3: return lm::log_gamma_p_ratio_mixed_d3(a, c).v[1];
        case 4: return lm::log_gamma_p_ratio_mixed_d4(a, c).v[1];
        case 5: return lm::log_gamma_p_ratio_mixed_d5(a, c).v[1];
        default: return lm::log_gamma_p_ratio_mixed_d6(a, c).v[1];
      }
    }
    switch (k) {
      case 1: return lm::log_gamma_q_ratio_mixed_d1(a, c).v[1];
      case 2: return lm::log_gamma_q_ratio_mixed_d2(a, c).v[1];
      case 3: return lm::log_gamma_q_ratio_mixed_d3(a, c).v[1];
      case 4: return lm::log_gamma_q_ratio_mixed_d4(a, c).v[1];
      case 5: return lm::log_gamma_q_ratio_mixed_d5(a, c).v[1];
      default: return lm::log_gamma_q_ratio_mixed_d6(a, c).v[1];
    }
  }
};

// the route of the chosen implementation on the host
std::string route_of(int k, double a, double c, bool p) {
  if (c == 0.0 || !std::isfinite(c)) return "limit";
  if (a < gr::a_fallback) return "fallback";
  gr::ratio_side s;
  if (gr::temme_region(a, c, false)) {
    if (gr::temme_ratio(a, c, k, p, s)) return s.direct ? "temme" : "temme+c";
  }
  const double x = a * c;
  const bool lower = gd::side_lower(a, x, p);
  const std::string cs = lower == p ? "" : "+c";
  if ((lower || x >= 36.0) && gr::series_ratio(a, c, lower, k, lower ? gd::nmax_lower_host : gd::nmax_upper_host, s))
    return (lower ? "lseries" : "useries") + cs;
  return (lower ? "quadL" : "quadU") + cs;
}

double ulp_of(double s) { return std::ldexp(1.0, std::ilogb(std::max(s, 2.2250738585072014e-308)) - 52); }

// the scale of each row (see the top comment)
std::vector<double> scales(int k, const bench::Ref& r, bool ratio) {
  std::map<std::pair<double, double>, double> m;
  auto key = [&](std::size_t i) {
    return ratio ? std::make_pair(r.x[i][0], r.x[i][1]) : std::make_pair(r.x[i][0], r.x[i][2]);
  };
  auto w = [&](std::size_t i) {
    if (!ratio) return 1.0;
    const int j = static_cast<int>(r.x[i][2]);
    return std::pow(r.x[i][0], k - j) * std::pow(r.x[i][1], j);
  };
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    auto& s = m[key(i)];
    s = std::max(s, std::fabs(r.hi[i]) * w(i));
  }
  std::vector<double> out(r.x.size());
  for (std::size_t i = 0; i < r.x.size(); ++i) out[i] = m[key(i)] / w(i);
  return out;
}

void breakdown(int k, bool p, const bench::Ref& r, const std::vector<double>& y, bool ratio) {
  struct St {
    int n = 0, bad = 0;
    double w[7] = {0, 0, 0, 0, 0, 0, 0};
    double o[7] = {0, 0, 0, 0, 0, 0, 0};
    bench::Args at[7]{};
  };
  const auto sc = scales(k, r, ratio);
  std::map<std::string, St> st;
  St all;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const int j = static_cast<int>(r.x[i][2]);
    const double a = r.x[i][0];
    std::string route =
        ratio ? (gr::temme_region(a, r.x[i][1], true) ? "Temme region" : route_of(k, a, r.x[i][1], p)) : "";
    if (ratio) route += a < 2.0 ? " a<2" : a < 20.0 ? " 2<=a<20" : a < 50.0 ? " 20<=a<50" : a < 100.0 ? " 50<=a<100" : a < 300.0 ? " 100<=a<300" : " a>=300";
    auto& g = st[route];
    if (j == 0) ++g.n, ++all.n;
    const double u = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / ulp_of(sc[i]);
    if (!std::isfinite(u)) {
      ++g.bad, ++all.bad;
      continue;
    }
    if (u > g.w[j]) g.w[j] = u, g.at[j] = r.x[i];
    if (u > all.w[j]) all.w[j] = u, all.at[j] = r.x[i];
    // the error in ulp of the component itself, where it is above 1e-12 of the scale
    if (std::fabs(r.hi[i]) > 1e-12 * sc[i]) {
      const double o = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / ulp_of(std::fabs(r.hi[i]));
      g.o[j] = std::max(g.o[j], o);
      all.o[j] = std::max(all.o[j], o);
    }
  }
  auto line = [&](const std::string& name, const St& g) {
    std::printf("    %-20s n %4d: max S by j:", name.c_str(), g.n);
    for (int j = 0; j <= k; ++j) std::printf(" %8.3g", g.w[j]);
    std::printf("%s\n", g.bad ? " NON-FINITE" : "");
    std::printf("    %-20s         max O by j:", "");
    for (int j = 0; j <= k; ++j) std::printf(" %8.3g", g.o[j]);
    std::printf("\n");
  };
  line("all", all);
  for (int j = 0; j <= k; ++j) std::printf("      worst j = %d at (%s)\n", j, bench::show(all.at[j], 2).c_str());
  if (ratio)
    for (const auto& [name, g] : st) line(name, g);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_gamma_ratio <reference dir> <output dir> [name filter] [impl filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  const std::string ifilter = argc > 4 ? argv[4] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 14;
  std::mt19937_64 gen(20261112);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> z(0.0, 1.0);
  // the timing set of math_gamma_mixed: a in [1/8, 1024], x = a e^(z/2), here c = e^(z/2)
  const auto t_ratio = bench::make_time_args(n, [&] {
    const double a = std::exp2(-3.0 + 13.0 * u01(gen));
    return bench::Args{a, std::exp(0.5 * z(gen)), 0};
  });
  const auto t_mixed = bench::make_time_args(n, [&] {
    const double a = std::exp2(-3.0 + 13.0 * u01(gen));
    return bench::Args{a, a * std::exp(0.5 * z(gen)), 0};
  });

  auto run = [&](const std::string& name, const char* impl, int k, bool p, auto f, bool ratio) {
    if (!filter.empty() && name.find(filter) == std::string::npos) return;
    if (!ifilter.empty() && std::string(impl).find(ifilter) == std::string::npos) return;
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    bench::report(name.c_str(), impl, r, f, outdir, ratio ? t_ratio : t_mixed);
    breakdown(k, p, r, bench::eval_kernel(r.x, f), ratio);
    std::fflush(stdout);
  };

  for (int k = 1; k <= 6; ++k) {
    for (int side = 0; side < 2; ++side) {
      const bool p = side == 0;
      const std::string s = p ? "p" : "q";
      const std::string name = "log_gamma_" + s + "_ratio_mixed_d" + std::to_string(k);
      run(name, "own", k, p, RatioF{k, p, 0}, true);
      run(name, "devroute", k, p, RatioF{k, p, 1}, true);
      run(name, "quad", k, p, RatioF{k, p, 2}, true);
      run(name, "noTemme", k, p, RatioF{k, p, 3}, true);
      run(name, "Temme20", k, p, RatioF{k, p, 4}, true);
      run(name, "fallback", k, p, RatioF{k, p, 7}, true);
      const std::string pname = "log_gamma_" + s + "_mixed_poisson_d" + std::to_string(k);
      run(pname, "own", k, p, MixedF{k, p, 0}, false);
      run(pname, "products", k, p, MixedF{k, p, 9}, false);
      run(pname, "noratio", k, p, MixedF{k, p, 4}, false);
      run(pname, "noTemme", k, p, MixedF{k, p, 3}, false);
      run(pname, "allTemme", k, p, MixedF{k, p, 10}, false);
      if ((ifilter.empty() || ifilter == "public") && (filter.empty() || name.find(filter) != std::string::npos)) {
        const double s1 = bench::time_ns<bench::HostOne>(t_ratio.host, PublicRatio{k, p}, 3);
        const double sx = bench::time_ns<bench::Exec>(t_ratio.dev, PublicRatio{k, p}, 5);
        std::printf("%s public: ns/call Serial %8.2f, %s %8.3f\n", name.c_str(), s1, bench::Exec::name(), sx);
      }
    }
  }
  return 0;
}
