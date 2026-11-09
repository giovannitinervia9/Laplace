// Accuracy and speed of the derivatives of log I_x(a, b) and log(1 - I_x(a, b))
// with respect to the shapes (include/laplace/math/incomplete_beta_deriv.hpp),
// orders 1 to 5, every mixed component. Candidates:
//   quad    the double-exponential quadrature (rays in log t and -log(1 - t))
//           everywhere;
//   series  the power series differentiated term by term where it converges
//           fast (x <= 1/2 and (b <= 1 or b x <= 0.7), or the same on the
//           upper side), the quadrature elsewhere (the choice of sides is in
//           detail_ibd::derivs);
//   fd      central differences of own::beta_inc's log I in a and b (a
//           baseline: what a stencil gives), step eps^(1/(K+2)) min(a, 1 + sqrt a)
//           in a and the same in b.
//
// Errors per component, against tests/data/reference/log_ibeta(c)_d{i}{j}.csv:
//   ulp   relative, in ulps of max(|f|, 2^-1022) (values below the double
//         range are stored as 0);
//   ulp1  in ulps of max(|f|, 1);
//   ulpN  in ulps of max(|f|, N_ij), N_ij = |f_K0|^(i/K) |f_0K|^(j/K) the
//         geometric mean of the pure components of the same order (by the
//         Cauchy-Schwarz bound of mixed cumulants, the scale of a mixed
//         component); equal to ulp where |f| >= N_ij;
//   ulpL  ulpN / max(1, |log |f||): beyond the error inherent in a value that
//         comes from exp(log), as where the side composed from lies in a far
//         tail (own::beta_inc returns I and 1 - I as exp of their logarithms).
// Regions: tail when min(I, 1 - I) < 1e-6, else body; shapes small when
// min(a, b) < 1, skew when min >= 1 and max/min >= 100, large when min >= 100
// and max/min < 100, mid otherwise. "ser" and "quad" in the last columns: rows
// where the series candidate uses the series, and the others.
//
// Usage: math_beta_deriv <reference dir> [mode] [candidate filter]
//   mode acc (default): accuracy; all: accuracy and ns/call; worst: accuracy
//   and the five worst rows per order and side; time: ns/call only.
#include "measure.hpp"

#include <laplace/math/incomplete_beta_deriv.hpp>

#include <map>
#include <random>

namespace {

namespace ibd = laplace::math::own::detail_ibd;
namespace ibx = laplace::math::own::detail_ibx;

struct Cand {
  int method;
  bool upper;
  ibd::quad_par qp;
  template <int K>
  KOKKOS_INLINE_FUNCTION void eval(double a, double b, double x, double* o) const {
    double d[ibd::NC];
    ibd::derivs<K>(a, b, x, upper, method, qp, d);
    for (int j = 0; j <= K; ++j) o[j] = d[ibd::ix(K - j, j)];
  }
};

// central differences of log I (or log(1 - I)) from own::beta_inc
struct FD {
  bool upper;
  KOKKOS_INLINE_FUNCTION double f(double a, double b, double x) const {
    const auto r = laplace::math::own::beta_inc(a, b, x);
    return upper ? r.log_ic : r.log_i;
  }
  template <int K>
  KOKKOS_INLINE_FUNCTION void eval(double a, double b, double x, double* o) const {
    const double e = Kokkos::pow(2.220446049250313e-16, 1.0 / (K + 2));
    const double ha = e * Kokkos::fmin(a, 1.0 + Kokkos::sqrt(a)), hb = e * Kokkos::fmin(b, 1.0 + Kokkos::sqrt(b));
    for (int j = 0; j <= K; ++j) {
      const int i = K - j;
      double s = 0.0;
      for (int k = 0; k <= i; ++k)
        for (int l = 0; l <= j; ++l) {
          const double w = ((k + l) % 2 ? -1.0 : 1.0) * ibd::binom(i, k) * ibd::binom(j, l);
          s += w * f(a + (0.5 * i - k) * ha, b + (0.5 * j - l) * hb, x);
        }
      o[j] = s / (Kokkos::pow(ha, i) * Kokkos::pow(hb, j));
    }
  }
};

template <int K, class C>
struct Sum {
  C c;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    double o[K + 1];
    c.template eval<K>(a, b, x, o);
    double s = 0.0;
    for (int j = 0; j <= K; ++j) s += o[j];
    return s;
  }
};

template <int K, class C>
std::vector<std::array<double, 6>> eval_rows(const std::vector<bench::Args>& x, C c) {
  const std::size_t n = x.size();
  const auto hx = bench::to_host_args(x);
  bench::DArgs dx("x", n);
  Kokkos::deep_copy(dx, hx);
  Kokkos::View<double* [6], Kokkos::LayoutRight, bench::Exec::memory_space> dy("y", n);
  Kokkos::parallel_for(
      "eval", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t r) {
        double o[K + 1];
        c.template eval<K>(dx(r, 0), dx(r, 1), dx(r, 2), o);
        for (int j = 0; j <= K; ++j) dy(r, j) = o[j];
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<std::array<double, 6>> y(n);
  for (std::size_t r = 0; r < n; ++r)
    for (int j = 0; j <= K; ++j) y[r][j] = hy(r, j);
  return y;
}

constexpr double tiny = 2.2250738585072014e-308;

// the rule's parameters, changed for measurement through the environment
// (LAPLACE_BD_H, _ALPHA_G, _ALPHA_E, _GSW, _CAP, _THI, _HTS, _TTS, _CUT, _KX, _PMONO)
ibd::quad_par qpar() {
  ibd::quad_par q;
  auto env = [](const char* n, double& v) {
    if (const char* s = std::getenv(n)) v = std::atof(s);
  };
  env("LAPLACE_BD_H", q.h);
  env("LAPLACE_BD_HMEAN", q.hmean);
  env("LAPLACE_BD_HLOW", q.hlow);
  env("LAPLACE_BD_XWIDE", q.x_wide);
  if (const char* o = std::getenv("LAPLACE_BD_PURECHECK")) q.pure_check = std::atoi(o) != 0;
  env("LAPLACE_BD_SERBELOW", q.series_below);
  env("LAPLACE_BD_GL16", q.gl_c16);
  env("LAPLACE_BD_GL20", q.gl_c20);
  env("LAPLACE_BD_GL24", q.gl_c24);
  env("LAPLACE_BD_GL32", q.gl_c32);
  env("LAPLACE_BD_ESTART", q.e_start);
  env("LAPLACE_BD_DROPLO", q.drop_lo);
  env("LAPLACE_BD_DROPHI", q.drop_hi);
  if (const char* o = std::getenv("LAPLACE_BD_OWN_ELEM")) q.own_elem = std::atoi(o) != 0;
  if (const char* o = std::getenv("LAPLACE_BD_STORE")) q.store = std::atoi(o) != 0;
  env("LAPLACE_BD_ALPHA_G", q.alpha_g);
  env("LAPLACE_BD_ALPHA_E", q.alpha_e);
  env("LAPLACE_BD_GSW", q.g_switch);
  env("LAPLACE_BD_CAP", q.cap);
  env("LAPLACE_BD_HTS", q.hts);
  env("LAPLACE_BD_THI", q.thi);
  env("LAPLACE_BD_TTS", q.tts);
  env("LAPLACE_BD_CUT", q.cut);
  env("LAPLACE_BD_KX", q.kx_fin);
  env("LAPLACE_BD_PMONO", q.p_mono);
  env("LAPLACE_BD_POLE", q.pole_min);
  env("LAPLACE_BD_PDIRECT", q.p_direct);
  env("LAPLACE_BD_OWN", q.own_min);
  return q;
}

// references: ref[side][i][j]
struct Refs {
  bench::Ref r[2][6][6];
  std::vector<int> region;  // 0..7: tail*4 + shape
  std::vector<bool> series_row;
};

const char* region_name[8] = {"body/small", "body/skew", "body/mid", "body/large",
                              "tail/small", "tail/skew", "tail/mid", "tail/large"};

Refs load(const std::string& dir) {
  Refs R;
  for (int side = 0; side < 2; ++side)
    for (int n = 1; n <= 5; ++n)
      for (int j = 0; j <= n; ++j) {
        const int i = n - j;
        const std::string name = std::string(side ? "log_ibetac_d" : "log_ibeta_d") + std::to_string(i) + std::to_string(j);
        R.r[side][i][j] = bench::read_ref(dir + "/" + name + ".csv");
        if (R.r[side][i][j].x != R.r[0][1][0].x) {
          std::fprintf(stderr, "%s: arguments differ from log_ibeta_d10\n", name.c_str());
          std::exit(2);
        }
      }
  for (const auto& a : R.r[0][1][0].x) {
    const auto bi = laplace::math::own::beta_inc(a[0], a[1], a[2]);
    const double mn = std::min(a[0], a[1]), mx = std::max(a[0], a[1]);
    const int shape = mn < 1.0 ? 0 : (mx / mn >= 100.0 ? 1 : (mn >= 100.0 ? 3 : 2));
    const bool tail = std::min(bi.i, bi.ic) < 1e-6;
    R.region.push_back(tail * 4 + shape);
    const double x = a[2], y = 1.0 - x;
    const double lx = x < 0.5 ? std::log(x) : std::log1p(-y), ly = x < 0.5 ? std::log1p(-x) : std::log(y);
    const ibx::args p{a[0], a[1], x, y, lx, ly, ibx::lambda_of(a[0], a[1], x), true, x >= 0.5};
    R.series_row.push_back(ibd::series_ok(p) || ibd::series_ok(ibx::swapped(p)));
  }
  return R;
}

struct Err {
  double ulp = 0, ulp1 = 0, ulpN = 0, ulpL = 0;
  bench::Args worst{0, 0, 0};
  int worst_j = 0;
  std::size_t bad = 0;
  double reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double ser = 0, quad = 0;
};

template <int K>
Err errors(const Refs& R, int side, const std::vector<std::array<double, 6>>& y, bool worst_rows) {
  Err e;
  const auto& base = R.r[side][1][0];
  struct Row {
    double eN, eU;
    std::size_t r;
    int j;
  };
  std::vector<Row> rows;
  for (std::size_t r = 0; r < base.x.size(); ++r) {
    const double fK0 = R.r[side][K][0].hi[r], f0K = R.r[side][0][K].hi[r];
    for (int j = 0; j <= K; ++j) {
      const int i = K - j;
      const auto& ref = R.r[side][i][j];
      const double hi = ref.hi[r], lo = ref.lo[r], v = y[r][j];
      if (!std::isfinite(v)) {
        ++e.bad;
        continue;
      }
      const double N = std::pow(std::fabs(fK0), double(i) / K) * std::pow(std::fabs(f0K), double(j) / K);
      const double d = std::fabs((v - hi) - lo);
      const double u = d / bench::ulp_of(std::max(std::fabs(hi), tiny));
      const double u1 = d / bench::ulp_of(std::max(std::fabs(hi), 1.0));
      const double uN = d / bench::ulp_of(std::max({std::fabs(hi), N, tiny}));
      if (u > e.ulp) {
        e.ulp = u;
        e.worst = base.x[r];
        e.worst_j = j;
      }
      e.ulp1 = std::max(e.ulp1, u1);
      e.ulpN = std::max(e.ulpN, uN);
      if (hi != 0.0) e.ulpL = std::max(e.ulpL, uN / std::max(1.0, std::fabs(std::log(std::fabs(hi)))));
      e.reg[R.region[r]] = std::max(e.reg[R.region[r]], uN);
      (R.series_row[r] ? e.ser : e.quad) = std::max(R.series_row[r] ? e.ser : e.quad, uN);
      if (worst_rows) rows.push_back({uN, std::getenv("LAPLACE_BD_SORT_L") && hi != 0.0 ? uN / std::max(1.0, std::fabs(std::log(std::fabs(hi)))) : u, r, j});
    }
  }
  if (worst_rows) {
    std::sort(rows.begin(), rows.end(), [](const Row& u, const Row& v) { return u.eN > v.eN; });
    if (std::getenv("LAPLACE_BD_SORT_ULP") || std::getenv("LAPLACE_BD_SORT_L"))
      std::sort(rows.begin(), rows.end(), [](const Row& u, const Row& v) { return u.eU > v.eU; });
    for (std::size_t k = 0; k < std::min<std::size_t>(rows.size(), 6); ++k) {
      const auto& w = rows[k];
      const auto& a = base.x[w.r];
      const auto bi = laplace::math::own::beta_inc(a[0], a[1], a[2]);
      std::printf("      ulpN %9.3g ulp %9.3g d%d%d  a %-12.6g b %-12.6g x %-12.6g  I %.3g 1-I %.3g  ref %.6g got %.6g %s\n", w.eN, w.eU,
                  K - w.j, w.j, a[0], a[1], a[2], bi.i, bi.ic, R.r[side][K - w.j][w.j].hi[w.r], y[w.r][w.j],
                  R.series_row[w.r] ? "(series row)" : "");
    }
  }
  return e;
}

void print_err(const char* cand, int side, int K, const Err& e) {
  std::printf("%-7s %-7s K=%d | max %9.3g ulp (d%d%d at %s), %8.3g ulp1, %8.3g ulpN, %8.3g ulpL | ser %8.3g quad %8.3g | bad %zu\n",
              cand, side ? "ibetac" : "ibeta", K, e.ulp, K - e.worst_j, e.worst_j, bench::show(e.worst, 3).c_str(), e.ulp1,
              e.ulpN, e.ulpL, e.ser, e.quad, e.bad);
  std::printf("        ulpN by region:");
  for (int k = 0; k < 8; ++k) std::printf(" %s %.3g%s", region_name[k], e.reg[k], k < 7 ? "," : "\n");
}

template <int K, class C>
void accuracy(const Refs& R, const char* name, int side, C c, bool worst) {
  const auto y = eval_rows<K>(R.r[0][1][0].x, c);
  const Err e = errors<K>(R, side, y, false);
  print_err(name, side, K, e);
  if (worst) errors<K>(R, side, y, true);
}

template <int K, class C>
void timing(const char* name, int side, C c, const bench::TimeArgs& t) {
  const Sum<K, C> f{c};
  const double t1 = bench::time_ns<bench::HostOne>(t.host, f, 3);
  const double td = bench::time_ns<bench::Exec>(t.dev, f, 5);
  std::printf("%-7s %-7s K=%d | ns/call Serial %9.1f, %s %8.2f\n", name, side ? "ibetac" : "ibeta", K, t1,
              bench::Exec::name(), td);
}

template <int K>
void run_order(const Refs& R, const std::string& mode, const std::string& filter, const bench::TimeArgs& t) {
  const bool acc = mode != "time", tim = mode == "all" || mode == "time", worst = mode == "worst";
  for (int side = 0; side < 2; ++side) {
    const bool up = side == 1;
    auto go = [&](const char* name, auto c) {
      if (!filter.empty() && filter != name) return;
      if (acc) accuracy<K>(R, name, side, c, worst);
      if (tim) timing<K>(name, side, c, t);
    };
    go("quad", Cand{0, up, qpar()});
    go("series", Cand{1, up, qpar()});
    go("fd", FD{up});
  }
}

// ---------------------------------------------------------------- profile

KOKKOS_INLINE_FUNCTION ibx::args frame_args(double a, double b, double x, bool swap) {
  const double y = 1.0 - x;
  const double lx = x < 0.5 ? Kokkos::log(x) : Kokkos::log1p(-y), ly = x < 0.5 ? Kokkos::log1p(-x) : Kokkos::log(y);
  const ibx::args p{a, b, x, y, lx, ly, ibx::lambda_of(a, b, x), true, x >= 0.5};
  return swap ? ibx::swapped(p) : p;
}

struct PBetaInc {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return laplace::math::own::beta_inc(a, b, x).log_i;
  }
};
template <int K>
struct PAssemble {  // the polygamma and log-beta terms alone
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    double kap[ibd::NC], l[ibd::NC], lt[ibd::NC];
    for (int i = 0; i < ibd::NC; ++i) kap[i] = 0.0;
    ibd::assemble<K>(a, b, x, 0.0, kap, l, lt, true);
    double s = 0;
    for (int i = 0; i < ibd::NC; ++i) s += l[i] + lt[i];
    return s;
  }
};
template <int K>
struct PCumulants {  // the moment-to-cumulant recursion alone
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    double mu[ibd::NC], kap[ibd::NC];
    for (int i = 0; i < ibd::NC; ++i) mu[i] = 1.0 / (i + a);
    mu[0] = 1.0;
    ibd::cumulants<K>(mu, kap);
    return kap[ibd::ix(K, 0)] + kap[ibd::ix(0, K)] + b + x;
  }
};
template <int K>
struct PComplement {  // the composition log(1 - S) alone
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    double lt[ibd::NC], o[ibd::NC];
    for (int i = 0; i < ibd::NC; ++i) lt[i] = 1.0 / (i + a);
    ibd::complement<K>(lt, b, x, Kokkos::log(x), o);
    return o[ibd::ix(K, 0)];
  }
};
template <int K>
struct PNodes {  // the node loop alone: integrate on the monotone side's pieces
  ibd::quad_par qp;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    const ibx::args p0 = frame_args(a, b, x, false);
    const ibx::args p = p0.lambda >= 0.0 ? p0 : ibx::swapped(p0);
    const double r = p.x / p.y;
    const ibd::ray R{p.a, p.b, r, (p.lambda + p.x) / p.y, 1.0, true};
    const ibd::piece pc = ibd::ray_piece(R, 0.0, 0.0, R.lamv, Kokkos::fabs(p.b - 1.0) * r * (1.0 + r), 1.0, -1e300, qp);
    double m[ibd::NC];
    for (int i = 0; i < ibd::NC; ++i) m[i] = 0.0;
    ibd::moment_sink<K> sk{m, 0.0, 0.0};
    ibd::integrate(&pc, 1, 1.0, qp, sk);
    return m[ibd::ix(K, 0)];
  }
};
template <int K>
struct PQuad {  // the quadrature on the monotone side
  ibd::quad_par qp;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    const ibx::args p = frame_args(a, b, x, false);
    const ibx::args f = p.lambda >= 0.0 ? p : ibx::swapped(p);
    double l[ibd::NC], lt[ibd::NC], logS;
    ibd::quad_lower<K>(f, qp, l, lt, true, logS);
    return l[ibd::ix(K, 0)];
  }
};
template <int K>
struct PSeries {  // the series on a side where it applies (time args are drawn there)
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    const ibx::args f = frame_args(a, b, x, false);
    double l[ibd::NC], lt[ibd::NC], logS;
    ibd::series_lower<K>(f, l, lt, true, logS);
    return l[ibd::ix(K, 0)];
  }
};

void profile() {
  const std::size_t n = std::size_t(1) << 13;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };
  auto near_mean = [&](double a, double b, double k) {
    const double s = a + b, m = a / s, sd = std::sqrt(a * b / (s * s * (s + 1)));
    return std::min(0.999, std::max(1e-3, m + k * sd * (2 * u01(gen) - 1)));
  };
  struct Reg {
    const char* name;
    bench::TimeArgs t;
  };
  std::vector<Reg> regs;
  regs.push_back({"all (timing set)", bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), u01(gen)}; })});
  {
    // the same draws sorted by (a in quarter-octave bins, b): neighbouring
    // rows have similar shapes, as in a fit
    std::vector<bench::Args> v(n);
    for (std::size_t i = 0; i < n; ++i)
      for (int k = 0; k < 3; ++k) v[i][k] = regs[0].t.host(i, k);
    std::sort(v.begin(), v.end(), [](const bench::Args& p, const bench::Args& q) {
      const int bp = static_cast<int>(std::floor(4.0 * std::log2(p[0]))), bq = static_cast<int>(std::floor(4.0 * std::log2(q[0])));
      return bp != bq ? bp < bq : p[1] < q[1];
    });
    std::size_t i = 0;
    regs.push_back({"all, sorted by (a, b)", bench::make_time_args(n, [&] { return v[i++]; })});
  }
  regs.push_back({"large, x near mean", bench::make_time_args(n, [&] {
                    const double a = expo(4, 8), b = expo(4, 8);
                    return bench::Args{a, b, near_mean(a, b, 3)};
                  })});
  regs.push_back({"moderate 1-16", bench::make_time_args(n, [&] { return bench::Args{expo(0, 4), expo(0, 4), u01(gen)}; })});
  regs.push_back({"one shape < 1", bench::make_time_args(n, [&] { return bench::Args{expo(-2, 0), expo(-2, 8), u01(gen)}; })});
  regs.push_back({"lower tail", bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), expo(-30, -10)}; })});
  const auto ser = bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 0), 0.5 * u01(gen)}; });
  const ibd::quad_par qp = qpar();
  for (const auto& r : regs) {
    // nodes per call (host)
    long cnt = 0;
    double d[ibd::NC];
    ibd::quad_par qc = qp;
    int c = 0;
    qc.count = &c;
    for (std::size_t i = 0; i < 2048; ++i) {
      c = 0;
      ibd::derivs<5>(r.t.host(i, 0), r.t.host(i, 1), r.t.host(i, 2), false, 1, qc, d);
      cnt += c;
    }
    auto tm = [&](const char* what, auto f) {
      std::printf("  %-34s Serial %8.1f  %s %8.2f\n", what, bench::time_ns<bench::HostOne>(r.t.host, f, 3), bench::Exec::name(),
                  bench::time_ns<bench::Exec>(r.t.dev, f, 5));
    };
    long cnt1 = 0;
    for (std::size_t i = 0; i < 2048; ++i) {
      c = 0;
      PNodes<1>{qc}(r.t.host(i, 0), r.t.host(i, 1), r.t.host(i, 2));
      cnt1 += c;
    }
    long sides = 0, nser = 0;
    int sd = 0;
    ibd::quad_par qs = qp;
    qs.sides = &sd;
    for (std::size_t i = 0; i < 2048; ++i) {
      sd = 0;
      ibd::derivs<5>(r.t.host(i, 0), r.t.host(i, 1), r.t.host(i, 2), false, 1, qs, d);
      sides += sd;
      const ibx::args p0 = frame_args(r.t.host(i, 0), r.t.host(i, 1), r.t.host(i, 2), false);
      nser += ibd::series_ok(p0) || ibd::series_ok(ibx::swapped(p0));
    }
    std::printf("%s: %.1f nodes per call (order 5), %.1f on one monotone ray (scale search included); %.3f sides per call;"
                " %.2f of rows with a series\n",
                r.name, cnt / 2048.0, cnt1 / 2048.0, sides / 2048.0, nser / 2048.0);
    tm("derivs K=1", Sum<1, Cand>{Cand{1, false, qp}});
    tm("derivs K=5", Sum<5, Cand>{Cand{1, false, qp}});
    tm("beta_inc", PBetaInc{});
    tm("assemble K=5 (polygamma terms)", PAssemble<5>{});
    tm("quad_lower K=1 (monotone side)", PQuad<1>{qp});
    tm("quad_lower K=5 (monotone side)", PQuad<5>{qp});
    tm("cumulants K=5", PCumulants<5>{});
    tm("complement K=1", PComplement<1>{});
    tm("complement K=5", PComplement<5>{});
    tm("ray scale + node loop K=1 (one ray)", PNodes<1>{qp});
    tm("ray scale + node loop K=5 (one ray)", PNodes<5>{qp});
  }
  {
    // elementary functions on the arguments of the nodes (d in [-5, 0], q in [0, 5])
    const auto te = bench::make_time_args(n, [&] { return bench::Args{-5.0 * u01(gen), 5.0 * u01(gen), 0.0}; });
    auto tm = [&](const char* what, auto f) {
      std::printf("  %-34s Serial %8.2f  %s %8.3f\n", what, bench::time_ns<bench::HostOne>(te.host, f, 5), bench::Exec::name(),
                  bench::time_ns<bench::Exec>(te.dev, f, 7));
    };
    std::printf("elementary functions, ns per call:\n");
    tm("Kokkos::exp", KOKKOS_LAMBDA(double d, double) { return Kokkos::exp(d); });
    tm("Kokkos::expm1", KOKKOS_LAMBDA(double d, double) { return Kokkos::expm1(d); });
    tm("Kokkos::log1p", KOKKOS_LAMBDA(double, double q) { return Kokkos::log1p(q); });
    tm("Kokkos::log", KOKKOS_LAMBDA(double, double q) { return Kokkos::log(q + 1.0); });
    tm("own::exp", KOKKOS_LAMBDA(double d, double) { return laplace::math::own::exp(d); });
    tm("own::log", KOKKOS_LAMBDA(double, double q) { return laplace::math::own::log(q + 1.0); });
    tm("detail_ibx::log1pmx (q/5)", KOKKOS_LAMBDA(double, double q) { return ibx::log1pmx(0.2 * q); });
    tm("ray_node", KOKKOS_LAMBDA(double d, double q) {
      const ibd::ray R{30.0, 20.0, q, q, 1.0, true};
      return ibd::ray_node(R, d).E;
    });
  }
  std::printf("series rows (x <= 1/2, b < 1):\n");
  std::printf("  series_lower K=1 Serial %8.1f\n", bench::time_ns<bench::HostOne>(ser.host, PSeries<1>{}, 3));
  std::printf("  series_lower K=5 Serial %8.1f  %s %8.2f\n", bench::time_ns<bench::HostOne>(ser.host, PSeries<5>{}, 3),
              bench::Exec::name(), bench::time_ns<bench::Exec>(ser.dev, PSeries<5>{}, 5));
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_beta_deriv <reference dir> [acc|all|worst|time] [candidate]\n");
    return 2;
  }
  const std::string refdir = argv[1];
  const std::string mode = argc > 2 ? argv[2] : "acc";
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  if (mode == "prof") {  // math_beta_deriv - prof: where the time goes, per region
    profile();
    return 0;
  }
  if (mode == "psi") {  // psi_all and dpsi_all against laplace::math::polygamma, relative differences
    std::mt19937_64 g(1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double worst[5] = {0, 0, 0, 0, 0};
    double ws[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 200000; ++i) {
      const double s = std::exp2(-7.0 + 20.0 * u(g));
      double P[5];
      ibd::psi_all(s, 4, P);
      for (int n = 1; n <= 4; ++n) {
        const double r = laplace::math::polygamma(n, s);
        const double e = std::fabs(P[n] - r) / bench::ulp_of(r);
        if (e > worst[n]) worst[n] = e, ws[n] = s;
      }
    }
    for (int n = 1; n <= 4; ++n) std::printf("psi_all n=%d: max %.3g ulp from polygamma at s = %.6g\n", n, worst[n], ws[n]);
    double wd[5] = {0, 0, 0, 0, 0}, wz[5] = {0, 0, 0, 0, 0}, wh[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 200000; ++i) {
      const double z = std::exp2(-7.0 + 19.0 * u(g)), h = std::exp2(-7.0 + 19.0 * u(g));
      double P[5], D;
      ibd::dpsi_all(z, h, 4, P, D);
      for (int n = 0; n <= 4; ++n) {
        const double r = n == 0 ? ibd::dD(z, h) : ibd::dpsi(n, z, h);
        const double e = std::fabs((n == 0 ? D : P[n]) - r) / bench::ulp_of(r);
        if (e > wd[n]) wd[n] = e, wz[n] = z, wh[n] = h;
      }
    }
    for (int n = 0; n <= 4; ++n)
      std::printf("dpsi_all n=%d (0: dD): max %.3g ulp from the separate one at z = %.6g, h = %.6g\n", n, wd[n], wz[n], wh[n]);
    return 0;
  }
  if (mode == "point") {  // math_beta_deriv - point a b x [upper]: every component, host, by candidate
    const double a = std::atof(argv[3]), b = std::atof(argv[4]), x = std::atof(argv[5]);
    const bool up = argc > 6 && std::atoi(argv[6]) != 0;
    ibd::quad_par qp;
    if (argc > 7) qp.h = std::atof(argv[7]);
    if (argc > 8) qp.cap = std::atof(argv[8]);
    if (argc > 9) qp.thi = std::atof(argv[9]);
    for (int method = 0; method < 2; ++method) {
      double d[ibd::NC];
      ibd::derivs<5>(a, b, x, up, method, qp, d);
      std::printf("%s:", method ? "series" : "quad");
      for (int k = 1; k < ibd::NC; ++k) std::printf(" %.17g", d[k]);
      std::printf("\n");
    }
    return 0;
  }
  const Refs R = load(refdir);
  // timing arguments: a, b = 2^U(-2, 8), x uniform
  const std::size_t n = std::size_t(1) << 14;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };
  const auto t = bench::make_time_args(n, [&] { return bench::Args{expo(-2, 8), expo(-2, 8), u01(gen)}; });
  run_order<1>(R, mode, filter, t);
  run_order<2>(R, mode, filter, t);
  run_order<3>(R, mode, filter, t);
  run_order<4>(R, mode, filter, t);
  run_order<5>(R, mode, filter, t);
  return 0;
}
