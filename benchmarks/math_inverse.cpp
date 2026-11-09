// Accuracy and speed of the inverse incomplete gamma and beta functions:
// Laplace's own (include/laplace/math/incomplete_inverse.hpp) against
// Boost.Math in double (the Laplace policy, the only route on the GPU).
//
// Usage: math_inverse <reference dir> <output dir> [name filter]
//
// For each reference file (tools/reference/incomplete.py and inverse.py) and
// candidate it prints
//   ulp     the forward error |x - x*| in ulps of x* (the exact inverse);
//   ulpC    the forward error in units of max(ulp(x*), dx), dx the change of
//           x* caused by one ulp of the input (p, q, log p or log q):
//           dx = ulp(input) / |d input / dx|. An inverse computed from a
//           forward function accurate to k ulps has ulpC of about k + 1/2,
//           whatever the conditioning (x moves 1/a ulps per ulp of p for a
//           small shape a, so ulp alone says little there);
//   the share with ulpC <= 1, the 99th percentile and the maximum of ulpC,
//   the worst arguments, the non-finite results (wrong infinities or NaN),
//   and ns/call on one host thread and on the default execution space.
// The results are written to <output dir>/<name>_<impl>.bin for
// tools/reference/backward_inverse.py (backward error at 200 bits).
#include "measure.hpp"

#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/incomplete_inverse.hpp>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <random>

namespace {

using laplace::math::detail::boost_policy;
namespace own = laplace::math::own;

// ----------------------------------------------------------------- candidates

struct GPOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double p) const { return own::gamma_p_inv(a, p); }
};
struct GQOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double q) const { return own::gamma_q_inv(a, q); }
};
struct GPLOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double l) const { return own::gamma_p_inv_logp(a, l); }
};
struct GQLOwn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double l) const { return own::gamma_q_inv_logq(a, l); }
};
// the beta inverses return x, or 1 - x when `complement`
struct BPOwn {
  bool complement = false;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double p) const {
    const auto r = own::ibeta_inv(a, b, p);
    return complement ? r.y : r.x;
  }
};
struct BQOwn {
  bool complement = false;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double q) const {
    const auto r = own::ibetac_inv(a, b, q);
    return complement ? r.y : r.x;
  }
};
struct BPLOwn {
  bool complement = false;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double l) const {
    const auto r = own::ibeta_inv_logp(a, b, l);
    return complement ? r.y : r.x;
  }
};
struct BQLOwn {
  bool complement = false;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double l) const {
    const auto r = own::ibetac_inv_logq(a, b, l);
    return complement ? r.y : r.x;
  }
};
// Boost's beta inverse can throw on the host even with the ignore_error
// policy (the Newton iteration inside Temme's method 2 raises an
// evaluation_error); a throw is counted as a NaN result. The device has no
// exceptions.
#define LAPLACE_BOOST_GUARD(expr)                                                \
  KOKKOS_IF_ON_HOST((try { return expr; } catch (...) { return std::nan(""); })) \
  KOKKOS_IF_ON_DEVICE((return expr;))

struct GPBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double p) const {
    LAPLACE_BOOST_GUARD(boost::math::gamma_p_inv(a, p, boost_policy()));
  }
};
struct GQBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double q) const {
    LAPLACE_BOOST_GUARD(boost::math::gamma_q_inv(a, q, boost_policy()));
  }
};
struct BPBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double p) const {
    LAPLACE_BOOST_GUARD(boost::math::ibeta_inv(a, b, p, boost_policy()));
  }
};
struct BQBoost {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double q) const {
    LAPLACE_BOOST_GUARD(boost::math::ibetac_inv(a, b, q, boost_policy()));
  }
};

// ----------------------------------------------------------------- conditioning

enum class Kind { gp, gq, gpl, gql, bp, bq, bpl, bql };

// |d input / dx| at x (y = 1 - x given separately for the beta), in double
// (a yardstick, not a reference)
double sensitivity(Kind k, const bench::Args& arg, double x, double y) {
  if (k == Kind::gp || k == Kind::gq || k == Kind::gpl || k == Kind::gql) {
    const double a = arg[0];
    const own::detail_inverse::gamma_shape sh(a);
    double l = sh.log_xdens(x) - std::log(x);  // log f(x)
    if (k == Kind::gpl) l -= own::gamma_inc(a, x).log_p;
    if (k == Kind::gql) l -= own::gamma_inc(a, x).log_q;
    return std::exp(l);
  }
  const double a = arg[0], b = arg[1];
  const own::detail_inverse::beta_shape sh(a, b);
  double l = sh.log_d(x, y) - std::log(x) - std::log(y);
  const auto r = x <= 0.5 ? own::beta_inc(a, b, x) : own::beta_inc(b, a, y);
  const double log_i = x <= 0.5 ? r.log_i : r.log_ic, log_ic = x <= 0.5 ? r.log_ic : r.log_i;
  if (k == Kind::bpl) l -= log_i;
  if (k == Kind::bql) l -= log_ic;
  return std::exp(l);
}

struct Stats {
  double max_ulp = 0, max_c = 0, q99_c = 0, share_c1 = 0;
  bench::Args worst{0, 0, 0};
  double worst_x = 0, worst_y = 0;
  std::size_t n_bad = 0;
  bench::Args bad{0, 0, 0};
};

// complement: r holds 1 - x* and y the computed 1 - x
Stats stats(Kind k, const bench::Ref& r, const std::vector<double>& y, bool complement = false) {
  Stats s;
  std::vector<double> cs;
  const int in = r.nargs - 1;  // index of the probability argument
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double xr = r.hi[i];
    const double e = bench::error_in(y[i], r.hi[i], r.lo[i], r.hi[i]);
    if (std::isinf(e)) {
      if (s.n_bad++ == 0) s.bad = r.x[i];
      continue;
    }
    const double err = std::fabs((y[i] - r.hi[i]) - r.lo[i]);
    const double px = complement ? 1.0 - xr : xr, py = complement ? xr : 1.0 - xr;
    const double dx = bench::ulp_of(r.x[i][in]) / sensitivity(k, r.x[i], px, py);
    const double c = err / std::fmax(bench::ulp_of(xr), std::isfinite(dx) ? dx : 0.0);
    s.max_ulp = std::max(s.max_ulp, std::fabs(e));
    cs.push_back(c);
    if (c > s.max_c) {
      s.max_c = c;
      s.worst = r.x[i];
      s.worst_x = xr;
      s.worst_y = y[i];
    }
  }
  if (!cs.empty()) {
    std::sort(cs.begin(), cs.end());
    s.q99_c = cs[std::min(cs.size() - 1, std::size_t(0.99 * cs.size()))];
    s.share_c1 = double(std::count_if(cs.begin(), cs.end(), [](double c) { return c <= 1.0; })) / cs.size();
  }
  return s;
}

// accuracy of 1 - x for the own beta inverses, on the rows with x* > 1/2
// (reference <name>_c.csv of tools/reference/inverse.py)
template <class F>
void report_complement(const char* name, Kind k, const bench::Ref& c, F f) {
  f.complement = true;
  const auto yk = bench::eval_kernel(c.x, f);
  const Stats sk = stats(k, c, yk, true);
  std::printf("%-17s %-5s | %s: 1-x on %zu rows with x* > 1/2: ulp max %9.3g | ulpC: <=1 %.4f, 99%% %8.3g, max %9.3g"
              " at (%s) 1-x* %.17g got %.17g | non-finite %zu\n",
              name, "own", bench::Exec::name(), c.x.size(), sk.max_ulp, sk.share_c1, sk.q99_c, sk.max_c,
              bench::show(sk.worst, c.nargs).c_str(), sk.worst_x, sk.worst_y, sk.n_bad);
  std::fflush(stdout);
}

template <class F>
void report(const char* name, const char* impl, Kind k, const bench::Ref& r, F f, const std::string& outdir,
            const bench::TimeArgs& t) {
  const auto yk = bench::eval_kernel(r.x, f);
  const auto yh = bench::eval_host(r.x, f);
  const Stats sk = stats(k, r, yk), sh = stats(k, r, yh);
  bench::write_bits(outdir + "/" + name + "_" + impl + ".bin", yk);
  const double t1 = bench::time_ns<bench::HostOne>(t.host, f, 3);
  const double tp = bench::time_ns<bench::Exec>(t.dev, f, 5);
  std::printf("%-17s %-5s | %s: ulp max %9.3g | ulpC: <=1 %.4f, 99%% %8.3g, max %9.3g at (%s) x* %.17g got %.17g"
              " | host ulpC max %9.3g | non-finite %zu%s%s%s | ns/call Serial %8.1f, %s %8.2f\n",
              name, impl, bench::Exec::name(), sk.max_ulp, sk.share_c1, sk.q99_c, sk.max_c,
              bench::show(sk.worst, r.nargs).c_str(), sk.worst_x, sk.worst_y, sh.max_c, sk.n_bad,
              sk.n_bad ? " (first at " : "", sk.n_bad ? bench::show(sk.bad, r.nargs).c_str() : "", sk.n_bad ? ")" : "",
              t1, bench::Exec::name(), tp);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_inverse <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 16;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };
  // gamma: a log-uniform in [2^-3, 2^10]; beta: a, b log-uniform in [2^-2, 2^8];
  // the probability uniform on (0, 1), or log-uniform down to 2^-1000 ("tail")
  const auto t_g = bench::make_time_args(n, [&] { return bench::Args{expo(-3, 10), u01(gen), 0}; });
  const auto t_gt = bench::make_time_args(n, [&] { return bench::Args{expo(-3, 10), expo(-1000, -1), 0}; });
  const auto t_gl = bench::make_time_args(n, [&] { return bench::Args{expo(-3, 10), std::log(u01(gen)), 0}; });
  const auto t_b = bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), u01(gen)}; });
  const auto t_bt = bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), expo(-1000, -1)}; });
  const auto t_bl =
      bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), std::log(u01(gen))}; });

  auto run = [&](const char* name, const char* impl, Kind k, auto f, const bench::TimeArgs& t) {
    if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    report(name, impl, k, r, f, outdir, t);
    if constexpr (requires { f.complement; })
      report_complement(name, k, bench::read_ref(refdir + "/" + name + "_c.csv"), f);
  };
  run("gamma_p_inv", "own", Kind::gp, GPOwn{}, t_g);
  run("gamma_p_inv", "boost", Kind::gp, GPBoost{}, t_g);
  run("gamma_p_inv_tail", "own", Kind::gp, GPOwn{}, t_gt);
  run("gamma_p_inv_tail", "boost", Kind::gp, GPBoost{}, t_gt);
  run("gamma_q_inv", "own", Kind::gq, GQOwn{}, t_g);
  run("gamma_q_inv", "boost", Kind::gq, GQBoost{}, t_g);
  run("gamma_q_inv_tail", "own", Kind::gq, GQOwn{}, t_gt);
  run("gamma_q_inv_tail", "boost", Kind::gq, GQBoost{}, t_gt);
  run("gamma_p_inv_logp", "own", Kind::gpl, GPLOwn{}, t_gl);
  run("gamma_q_inv_logq", "own", Kind::gql, GQLOwn{}, t_gl);
  run("ibeta_inv", "own", Kind::bp, BPOwn{}, t_b);
  run("ibeta_inv", "boost", Kind::bp, BPBoost{}, t_b);
  run("ibeta_inv_tail", "own", Kind::bp, BPOwn{}, t_bt);
  run("ibeta_inv_tail", "boost", Kind::bp, BPBoost{}, t_bt);
  run("ibetac_inv", "own", Kind::bq, BQOwn{}, t_b);
  run("ibetac_inv", "boost", Kind::bq, BQBoost{}, t_b);
  run("ibetac_inv_tail", "own", Kind::bq, BQOwn{}, t_bt);
  run("ibetac_inv_tail", "boost", Kind::bq, BQBoost{}, t_bt);
  run("ibeta_inv_logp", "own", Kind::bpl, BPLOwn{}, t_bl);
  run("ibetac_inv_logq", "own", Kind::bql, BQLOwn{}, t_bl);
  return 0;
}
