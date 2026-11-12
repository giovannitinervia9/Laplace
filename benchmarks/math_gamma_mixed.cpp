// Accuracy and speed of the mixed partial derivatives of log P(a, x) and
// log Q(a, x) in a and x, total orders 1 to 6
// (include/laplace/math/incomplete_gamma_mixed.hpp).
//
// Candidates: the chosen route (own); the chosen route with the series limits
// of the device, run on the host (devroute: what the CUDA build computes);
// the quadrature alone (quad); the chosen route without Temme's expansion
// (noTemme); the chosen route without the ratio S(h, e)/S through the density
// (noratio: every series in e from the cumulants) and with it for every
// complement (allratio); log Q for a <= 1, x < 3/2 by the quadrature of the
// upper side alone (uquad); the chosen route without the quadrature for the
// component j = 1 of log Q at a < 2, 1/4 <= x < 3/2 (noj1quad); and, for
// orders 1 and 2, the formulas in the hazard
// H = f/S formed from the library's outputs (naive):
//   d log P/dx = H,  d^2 log P/da dx = H (log x - psi(a) - d log P/da),
//   d^2 log P/dx^2 = H ((a - 1)/x - 1 - H)  (Q: -H and +H),
// with H = exp(log f - log S), which cancel where the new route does not.
//
// The reference files log_gamma_{p,q}_mixed_d{K}.csv hold (a, x, j) and
// d^K / da^(K-j) dx^j. Each report line covers all components; it is followed
// by the largest min(ulpL, ulp1) by component j, by the route the chosen
// implementation takes ("+c": the complement log(1 - S)) and by the range of a
// ("a>=1 int": the integer shapes of the Poisson cdf).
//
// Usage: math_gamma_mixed <reference dir> <output dir> [name filter] [impl filter]
#include "measure.hpp"

#include <laplace/math/incomplete_gamma_mixed.hpp>

#include <map>
#include <random>
#include <type_traits>

namespace {

namespace gd = laplace::math::own::detail_gamma_deriv;
namespace gm = laplace::math::own::detail_gamma_mixed;

struct Mixed {
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
struct Public {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x, double) const {
    namespace lm = laplace::math;
    if (p) {
      switch (k) {
        case 1: return lm::log_gamma_p_mixed_d1(a, x).v[1];
        case 2: return lm::log_gamma_p_mixed_d2(a, x).v[1];
        case 3: return lm::log_gamma_p_mixed_d3(a, x).v[1];
        case 4: return lm::log_gamma_p_mixed_d4(a, x).v[1];
        case 5: return lm::log_gamma_p_mixed_d5(a, x).v[1];
        default: return lm::log_gamma_p_mixed_d6(a, x).v[1];
      }
    }
    switch (k) {
      case 1: return lm::log_gamma_q_mixed_d1(a, x).v[1];
      case 2: return lm::log_gamma_q_mixed_d2(a, x).v[1];
      case 3: return lm::log_gamma_q_mixed_d3(a, x).v[1];
      case 4: return lm::log_gamma_q_mixed_d4(a, x).v[1];
      case 5: return lm::log_gamma_q_mixed_d5(a, x).v[1];
      default: return lm::log_gamma_q_mixed_d6(a, x).v[1];
    }
  }
};

struct Naive {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x, double jd) const {
    namespace lm = laplace::math;
    const int j = static_cast<int>(jd);
    const auto r = lm::own::gamma_inc(a, x);
    const double ls = p ? r.log_p : r.log_q;
    const double lf = (a - 1.0) * lm::log(x) - x - lm::lgamma(a);
    const double H = lm::exp(lf - ls);
    const double sg = p ? 1.0 : -1.0;
    if (k == 1) return j == 0 ? (p ? lm::log_gamma_p_da1(a, x) : lm::log_gamma_q_da1(a, x)) : sg * H;
    if (j == 0) return p ? lm::log_gamma_p_da2(a, x) : lm::log_gamma_q_da2(a, x);
    if (j == 1) {
      const double d1 = p ? lm::log_gamma_p_da1(a, x) : lm::log_gamma_q_da1(a, x);
      return sg * H * (lm::log(x) - lm::digamma(a) - d1);
    }
    return sg * H * ((a - 1.0) / x - 1.0 - sg * H);
  }
};

// the route of the chosen implementation (the dispatch of log_gamma_mixed_impl on the host)
std::string route_of(int k, double a, double x, bool p, bool device) {
  if (x == 0.0 || !std::isfinite(x)) return "limit";
  gm::mixed_side s;
  const double sg = std::fabs(x - a) / a;
  if ((a >= 20.0 && sg < 0.3) || (a >= 300.0 && sg < 0.4) || (a >= 1000.0 && sg < 0.5)) {
    if (gm::temme_mixed(a, x, k, p, s)) return s.direct ? "temme" : "temme+c";
  }
  const bool lower = gd::side_lower(a, x, p);
  const bool direct = lower == p;
  const std::string c = direct ? "" : "+c";
  int nl = gd::nmax_lower_host, nu = gd::nmax_upper_host;
  if (device) nl = a < 2.0 ? gd::nmax_lower_device : 0, nu = gd::nmax_upper_device;
  if (direct && !lower && x < 1.5 && a < 2.0 && gm::small_x_mixed(a, x, k, s)) return "smallx" + c;
  if ((lower || x >= 36.0) && gm::series_mixed(a, x, lower, direct, k, lower ? nl : nu, s))
    return (lower ? "lseries" : "useries") + c;
  return (lower ? "quadL" : "quadU") + c;
}

double min_err(double y, double hi, double lo) {
  const double e = std::fabs(bench::error_in(y, hi, lo, std::max(std::fabs(hi), 2.2250738585072014e-308)));
  if (!std::isfinite(e)) return INFINITY;
  const double e1 = std::fabs(bench::error_in(y, hi, lo, std::max(std::fabs(hi), 1.0)));
  const double eL = e / std::max(1.0, std::fabs(std::log(std::fabs(hi))));
  return std::min(e1, eL);
}

void breakdown(int k, bool p, const bench::Ref& r, const std::vector<double>& y, bool device) {
  struct St {
    int n = 0, bad = 0;
    double w[7] = {0, 0, 0, 0, 0, 0, 0};
    bench::Args at[7]{};
  };
  std::map<std::string, St> st, by_a;
  St all;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const int j = static_cast<int>(r.x[i][2]);
    const double a = r.x[i][0];
    auto& g = st[route_of(k, a, r.x[i][1], p, device)];
    auto& ga = by_a[a < 0.1 ? "a<0.1" : a < 1.0 ? "0.1<=a<1" : a == std::floor(a) ? "a>=1 int" : "a>=1"];
    if (j == 0) ++g.n, ++all.n, ++ga.n;
    const double u = min_err(y[i], r.hi[i], r.lo[i]);
    if (!std::isfinite(u)) {
      ++g.bad, ++all.bad, ++ga.bad;
      continue;
    }
    if (u > g.w[j]) g.w[j] = u, g.at[j] = r.x[i];
    if (u > ga.w[j]) ga.w[j] = u, ga.at[j] = r.x[i];
    if (u > all.w[j]) all.w[j] = u, all.at[j] = r.x[i];
  }
  auto line = [&](const std::string& name, const St& g) {
    std::printf("    %-11s n %4d: max min(ulpL, ulp1) by j:", name.c_str(), g.n);
    for (int j = 0; j <= k; ++j) std::printf(" %8.3g", g.w[j]);
    std::printf("%s\n", g.bad ? " NON-FINITE" : "");
  };
  line("all", all);
  for (int j = 0; j <= k; ++j)
    std::printf("      worst j = %d at (%s)\n", j, bench::show(all.at[j], 2).c_str());
  for (const auto& [name, g] : st) line(name, g);
  for (const auto& [name, g] : by_a) line(name, g);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_gamma_mixed <reference dir> <output dir> [name filter] [impl filter]\n");
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
  // the timing set of math_gamma_deriv: a in [1/8, 1024], x = a e^(z/2)
  const auto t_gamma = bench::make_time_args(n, [&] {
    const double a = std::exp2(-3.0 + 13.0 * u01(gen));
    return bench::Args{a, a * std::exp(0.5 * z(gen)), 0};
  });

  auto run = [&](const std::string& name, const char* impl, int k, bool p, auto f, bool device_route) {
    if (!filter.empty() && name.find(filter) == std::string::npos) return;
    if (!ifilter.empty() && std::string(impl).find(ifilter) == std::string::npos) return;
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    bench::report(name.c_str(), impl, r, f, outdir, t_gamma);
    breakdown(k, p, r, bench::eval_kernel(r.x, f), device_route);
    std::fflush(stdout);
  };

  for (int k = 1; k <= 6; ++k) {
    for (int side = 0; side < 2; ++side) {
      const bool p = side == 0;
      const std::string name = std::string(p ? "log_gamma_p_mixed_d" : "log_gamma_q_mixed_d") + std::to_string(k);
      run(name, "own", k, p, Mixed{k, p, 0}, false);
      run(name, "devroute", k, p, Mixed{k, p, 1}, true);
      run(name, "quad", k, p, Mixed{k, p, 2}, false);
      run(name, "noTemme", k, p, Mixed{k, p, 3}, false);
      run(name, "noratio", k, p, Mixed{k, p, 4}, false);
      run(name, "allratio", k, p, Mixed{k, p, 5}, false);
      run(name, "uquad", k, p, Mixed{k, p, 6}, false);
      run(name, "noj1quad", k, p, Mixed{k, p, 8}, false);
      if (k <= 2) run(name, "naive", k, p, Naive{k, p}, false);
      if (ifilter.empty() || ifilter == "public") {
        if (filter.empty() || name.find(filter) != std::string::npos) {
          const double s1 = bench::time_ns<bench::HostOne>(t_gamma.host, Public{k, p}, 3);
          const double sx = bench::time_ns<bench::Exec>(t_gamma.dev, Public{k, p}, 5);
          std::printf("%s public: ns/call Serial %8.2f, %s %8.3f\n", name.c_str(), s1, bench::Exec::name(), sx);
        }
      }
    }
  }
  return 0;
}
