// Accuracy and speed of log T_nu(z), the Student t distribution function with
// real degrees of freedom, and of its derivatives in (z, nu), orders 1 to 5,
// every mixed component (include/laplace/math/student.hpp). Candidates:
//   quad    integrals of derivatives of the density (centre: Gauss-Legendre in
//           s = sqrt(nu) sinh tau; lower tail: Takahasi-Mori nodes after the shift
//           s = z - u where z^2 < nu, after the scaling s = z e^v elsewhere;
//           upper side: composition of log(1 - S));
//   beta    composition from the shape derivatives of log I_x(nu/2, 1/2)
//           (log_ibeta_dK, log_ibetac_dK) and the x-derivatives
//           exp(log density - log I);
//   fd      central differences of own::log_student_cdf in z and nu (a
//           baseline), steps eps^(1/(K+2)) max(1, |z|) and eps^(1/(K+2)) min(nu, 1 + sqrt nu).
// The value: quad (log_cdf_quad) against own::beta_inc (log_cdf_beta).
//
// Errors per component against tests/data/reference/log_student_cdf_d{i}{j}.csv,
// as in benchmarks/math_beta_deriv.cpp: ulp (relative, floored at 2^-1022), ulp1
// (in ulps of max(|f|, 1)), ulpN (in ulps of max(|f|, N_ij), N_ij the
// geometric mean of the pure components of the order), ulpL = ulpN / max(1, |log |f||).
// Regions (ulpN): |z| <= 6, z < -6 (lower tail), z > 6 (upper tail), and across
// them nu < 1 and nu > 1000.
//
// Usage: math_student <reference dir> [mode] [candidate]
//   mode acc (default), all (with ns/call), worst (five worst rows per order),
//   time, prof (nodes per call by region), deep (the deep tail without
//   references: candidates against each other), point z nu.
#include "measure.hpp"

#include <laplace/math/student.hpp>

#include <functional>
#include <map>
#include <random>

namespace {

namespace st = laplace::math::own::detail_st;
namespace ibd = laplace::math::own::detail_ibd;
constexpr int NC = st::NC;

st::par qpar() {
  st::par q;
  auto env = [](const char* n, double& v) {
    if (const char* s = std::getenv(n)) v = std::atof(s);
  };
  env("LAPLACE_ST_HG", q.h_g);
  env("LAPLACE_ST_AG", q.alpha_g);
  env("LAPLACE_ST_AP", q.alpha_p);
  env("LAPLACE_ST_START", q.start);
  env("LAPLACE_ST_SIGG", q.sig_g);
  env("LAPLACE_ST_HP", q.h_p);
  env("LAPLACE_ST_SIGP", q.sig_p);
  env("LAPLACE_ST_TOL", q.tol);
  env("LAPLACE_ST_PIECE0", q.piece0);
  env("LAPLACE_ST_SWITCH", q.t_switch);
  env("LAPLACE_ST_SHIFT", q.shift_ratio);
  env("LAPLACE_ST_NUW", q.nu_w);
  if (const char* s = std::getenv("LAPLACE_ST_GLN")) q.gl_n = std::atoi(s);
  return q;
}

struct Cand {
  int method;  // 0 quad, 1 beta, 2 fd
  st::par pp;
  template <int K>
  KOKKOS_INLINE_FUNCTION void eval(double z, double nu, double* o) const {
    double L[NC];
    if (method == 0) st::derivs<K>(z, nu, pp, L);
    else if (method == 1) st::derivs_beta<K>(z, nu, L);
    else {
      fd<K>(z, nu, o);
      return;
    }
    const auto r = st::pick<K>(L);
    for (int j = 0; j <= K; ++j) o[j] = r.c[j];
  }
  template <int K>
  KOKKOS_INLINE_FUNCTION void fd(double z, double nu, double* o) const {
    const double e = Kokkos::pow(2.220446049250313e-16, 1.0 / (K + 2));
    const double hz = e * Kokkos::fmax(1.0, Kokkos::fabs(z)), hn = e * Kokkos::fmin(nu, 1.0 + Kokkos::sqrt(nu));
    for (int j = 0; j <= K; ++j) {
      const int i = K - j;
      double s = 0.0;
      for (int k = 0; k <= i; ++k)
        for (int l = 0; l <= j; ++l) {
          const double w = ((k + l) % 2 ? -1.0 : 1.0) * ibd::binom(i, k) * ibd::binom(j, l);
          s += w * st::log_cdf_quad(z + (0.5 * i - k) * hz, nu + (0.5 * j - l) * hn);
        }
      o[j] = s / (Kokkos::pow(hz, i) * Kokkos::pow(hn, j));
    }
  }
};

template <int K, class C>
struct Sum {
  C c;
  KOKKOS_INLINE_FUNCTION double operator()(double z, double nu) const {
    double o[K + 1];
    c.template eval<K>(z, nu, o);
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
        c.template eval<K>(dx(r, 0), dx(r, 1), o);
        for (int j = 0; j <= K; ++j) dy(r, j) = o[j];
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<std::array<double, 6>> y(n);
  for (std::size_t r = 0; r < n; ++r)
    for (int j = 0; j <= K; ++j) y[r][j] = hy(r, j);
  return y;
}

constexpr double tiny = 2.2250738585072014e-308;

struct Refs {
  bench::Ref v;          // the value
  bench::Ref r[6][6];    // r[i][j]
  std::vector<int> reg;  // 0 centre, 1 lower, 2 upper
};

Refs load(const std::string& dir) {
  Refs R;
  R.v = bench::read_ref(dir + "/log_student_cdf.csv");
  for (int n = 1; n <= 5; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      const std::string name = "log_student_cdf_d" + std::to_string(i) + std::to_string(j);
      R.r[i][j] = bench::read_ref(dir + "/" + name + ".csv");
      if (R.r[i][j].x != R.v.x) {
        std::fprintf(stderr, "%s: arguments differ\n", name.c_str());
        std::exit(2);
      }
    }
  for (std::size_t k = 0; k < R.v.x.size(); ++k) {
    const double z = R.v.x[k][0];
    // by value (the thirds of the draw overlap in [-6, 6])
    const int g = z < -6.0 ? 1 : (z > 6.0 ? 2 : 0);
    R.reg.push_back(g);
  }
  return R;
}

const char* reg_name[5] = {"|z|<=6", "z<-6", "z>6", "nu<1", "nu>1000"};

struct Err {
  double ulp = 0, ulp1 = 0, ulpN = 0, ulpL = 0;
  bench::Args worst{0, 0, 0};
  int worst_j = 0;
  std::size_t bad = 0;
  double reg[5] = {0, 0, 0, 0, 0};
  double nz = 0, nzL = 0;  // ulpN and ulpL without the pure z-components of orders 3, 5 at |z| < 1 (as the test)
};

template <int K>
Err errors(const Refs& R, const std::vector<std::array<double, 6>>& y, bool worst_rows) {
  Err e;
  struct Row {
    double eN, u;
    std::size_t r;
    int j;
  };
  std::vector<Row> rows;
  for (std::size_t r = 0; r < R.v.x.size(); ++r) {
    const double fK0 = R.r[K][0].hi[r], f0K = R.r[0][K].hi[r], nu = R.v.x[r][1];
    for (int j = 0; j <= K; ++j) {
      const int i = K - j;
      const auto& ref = R.r[i][j];
      const double hi = ref.hi[r], lo = ref.lo[r], v = y[r][j];
      if (!std::isfinite(v)) {
        ++e.bad;
        if (worst_rows) rows.push_back({1e300, 1e300, r, j});
        continue;
      }
      const double N = std::pow(std::fabs(fK0), double(i) / K) * std::pow(std::fabs(f0K), double(j) / K);
      const double d = std::fabs((v - hi) - lo);
      const double u = d / bench::ulp_of(std::max(std::fabs(hi), tiny));
      const double u1 = d / bench::ulp_of(std::max(std::fabs(hi), 1.0));
      const double uN = d / bench::ulp_of(std::max({std::fabs(hi), N, tiny}));
      if (u > e.ulp) {
        e.ulp = u;
        e.worst = R.v.x[r];
        e.worst_j = j;
      }
      e.ulp1 = std::max(e.ulp1, u1);
      e.ulpN = std::max(e.ulpN, uN);
      if (!(j == 0 && (K == 3 || K == 5) && std::fabs(R.v.x[r][0]) < 1.0)) {
        e.nz = std::max(e.nz, uN);
        if (hi != 0.0) e.nzL = std::max(e.nzL, uN / std::max(1.0, std::fabs(std::log(std::fabs(hi)))));
      }
      if (hi != 0.0) e.ulpL = std::max(e.ulpL, uN / std::max(1.0, std::fabs(std::log(std::fabs(hi)))));
      e.reg[R.reg[r]] = std::max(e.reg[R.reg[r]], uN);
      if (nu < 1.0) e.reg[3] = std::max(e.reg[3], uN);
      if (nu > 1000.0) e.reg[4] = std::max(e.reg[4], uN);
      if (worst_rows) rows.push_back({uN, u, r, j});
    }
  }
  if (worst_rows) {
    std::sort(rows.begin(), rows.end(), [](const Row& u, const Row& v) { return u.eN > v.eN; });
    for (std::size_t k = 0; k < std::min<std::size_t>(rows.size(), 6); ++k) {
      const auto& w = rows[k];
      const auto& a = R.v.x[w.r];
      std::printf("      ulpN %9.3g ulp %9.3g d%d%d  z %-13.7g nu %-13.7g  logT %.4g  ref %.10g got %.10g\n", w.eN, w.u,
                  K - w.j, w.j, a[0], a[1], R.v.hi[w.r], R.r[K - w.j][w.j].hi[w.r], y[w.r][w.j]);
    }
  }
  return e;
}

void print_err(const char* cand, int K, const Err& e) {
  std::printf("%-6s K=%d | max %9.3g ulp (d%d%d at %s), %8.3g ulp1, %8.3g ulpN, %8.3g ulpL | bad %zu\n", cand, K, e.ulp,
              K - e.worst_j, e.worst_j, bench::show(e.worst, 2).c_str(), e.ulp1, e.ulpN, e.ulpL, e.bad);
  std::printf("       as the test: %.3g ulpN, %.3g ulpL; ulpN by region:", e.nz, e.nzL);
  for (int k = 0; k < 5; ++k) std::printf(" %s %.3g%s", reg_name[k], e.reg[k], k < 4 ? "," : "\n");
}

template <int K, class C>
void timing(const char* name, C c, const bench::TimeArgs& t) {
  const Sum<K, C> f{c};
  const double t1 = bench::time_ns<bench::HostOne>(t.host, f, 3);
  const double td = bench::time_ns<bench::Exec>(t.dev, f, 5);
  std::printf("%-6s K=%d | ns/call Serial %9.1f, %s %8.2f\n", name, K, t1, bench::Exec::name(), td);
}

template <int K>
void run_order(const Refs& R, const std::string& mode, const std::string& filter, const bench::TimeArgs& t) {
  const bool acc = mode != "time", tim = mode == "all" || mode == "time", worst = mode == "worst";
  auto go = [&](const char* name, Cand c) {
    if (!filter.empty() && filter != name) return;
    if (acc) {
      const auto y = eval_rows<K>(R.v.x, c);
      print_err(name, K, errors<K>(R, y, false));
      if (worst) errors<K>(R, y, true);
    }
    if (tim) timing<K>(name, c, t);
  };
  go("quad", Cand{0, qpar()});
  go("beta", Cand{1, qpar()});
  if (mode != "worst") go("fd", Cand{2, qpar()});
}

// the value
struct VQuad {
  KOKKOS_INLINE_FUNCTION double operator()(double z, double nu) const { return st::log_cdf_quad(z, nu); }
};
struct VBeta {
  KOKKOS_INLINE_FUNCTION double operator()(double z, double nu) const { return st::log_cdf_beta(z, nu); }
};

void value(const Refs& R, const bench::TimeArgs& t, const std::string& outdir, const std::string& mode) {
  bench::report("log_student_cdf", "quad", R.v, VQuad{}, outdir, t);
  bench::report("log_student_cdf", "beta_inc", R.v, VBeta{}, outdir, t);
  if (mode == "worst") {
    for (int m = 0; m < 2; ++m) {
      std::vector<std::pair<double, std::size_t>> rows;
      for (std::size_t r = 0; r < R.v.x.size(); ++r) {
        const double z = R.v.x[r][0], nu = R.v.x[r][1];
        const double y = m == 0 ? st::log_cdf_quad(z, nu) : st::log_cdf_beta(z, nu);
        rows.push_back({std::fabs(bench::error_in(y, R.v.hi[r], R.v.lo[r], std::max(std::fabs(R.v.hi[r]), tiny))), r});
      }
      std::sort(rows.rbegin(), rows.rend());
      for (int k = 0; k < 5; ++k)
        std::printf("      %s value: %9.3g ulp at z %.7g nu %.7g (log T %.6g)\n", m ? "beta_inc" : "quad", rows[k].first,
                    R.v.x[rows[k].second][0], R.v.x[rows[k].second][1], R.v.hi[rows[k].second]);
    }
  }
}

// ---------------------------------------------------------------- deep tail

// |z| in [25, 2^21], nu in [4600, 1e6] (the rows mpmath could not certify):
// quad against quad with h/2 and a smaller tolerance, and against beta
template <int K>
void deep_order(const std::vector<bench::Args>& x) {
  st::par fine;
  fine.h_g /= 2;
  fine.h_p /= 2;
  fine.tol = 1e-22;
  fine.start = 60.0;
  const auto q = eval_rows<K>(x, Cand{0, st::par{}});
  const auto f = eval_rows<K>(x, Cand{0, fine});
  const auto b = eval_rows<K>(x, Cand{1, st::par{}});
  double eqf = 0, eqb = 0;
  bench::Args wf{0, 0, 0}, wb{0, 0, 0};
  std::size_t nb = 0;
  for (std::size_t r = 0; r < x.size(); ++r) {
    for (int j = 0; j <= K; ++j) {
      const double Nij = std::pow(std::fabs(q[r][0]), double(K - j) / K) * std::pow(std::fabs(q[r][K]), double(j) / K);
      const double sc = bench::ulp_of(std::max({std::fabs(f[r][j]), Nij, tiny}));
      if (!std::isfinite(q[r][j]) || !std::isfinite(f[r][j])) {
        ++nb;
        continue;
      }
      const double d1 = std::fabs(q[r][j] - f[r][j]) / sc, d2 = std::fabs(q[r][j] - b[r][j]) / sc;
      if (d1 > eqf) eqf = d1, wf = x[r];
      if (std::isfinite(d2) && d2 > eqb) eqb = d2, wb = x[r];
    }
  }
  std::printf("deep K=%d: quad vs fine quad max %.3g ulpN (at %s); quad vs beta max %.3g ulpN (at %s); non-finite %zu\n", K,
              eqf, bench::show(wf, 2).c_str(), eqb, bench::show(wb, 2).c_str(), nb);
}

void deep() {
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<bench::Args> x;
  for (int i = 0; i < 2000; ++i) {
    const double a = std::exp2(std::log2(25.0) + (21.0 - std::log2(25.0)) * u01(gen));
    const double nu = std::exp(std::log(4600.0) + (std::log(1e6) - std::log(4600.0)) * u01(gen));
    x.push_back({i % 2 ? a : -a, nu, 0.0});
  }
  // the value: quad against beta_inc
  for (int side = 0; side < 2; ++side) {
    double ev = 0;
    bench::Args wv{0, 0, 0};
    for (const auto& a : x) {
      if ((a[0] > 0.0) != (side == 1)) continue;
      const double q = st::log_cdf_quad(a[0], a[1]), b = st::log_cdf_beta(a[0], a[1]);
      const double d = std::fabs(q - b) / bench::ulp_of(std::max(std::fabs(b), tiny));
      if (d > ev) ev = d, wv = a;
    }
    std::printf("deep value, %s: quad vs beta_inc max %.3g ulp (at %s)\n", side ? "z > 0" : "z < 0", ev,
                bench::show(wv, 2).c_str());
  }
  deep_order<1>(x);
  deep_order<2>(x);
  deep_order<3>(x);
  deep_order<4>(x);
  deep_order<5>(x);
}

// ---------------------------------------------------------------- profile

void profile() {
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> nz(0.0, 1.0);
  struct Reg {
    const char* name;
    std::function<bench::Args()> draw;
  };
  auto expo = [&](double lo, double hi) { return std::exp2(lo + (hi - lo) * u01(gen)); };
  std::vector<Reg> regs = {
      {"timing set", [&] { return bench::Args{3.0 * nz(gen), expo(-2, 10), 0}; }},
      {"centre |z| < 0.5", [&] { return bench::Args{u01(gen) - 0.5, expo(-2, 10), 0}; }},
      {"Gaussian tail nu > 100", [&] { return bench::Args{-2.0 - 4.0 * u01(gen), expo(7, 12), 0}; }},
      {"power tail nu < 1", [&] { return bench::Args{-expo(4, 20), expo(-4, 0), 0}; }},
      {"small nu centre", [&] { return bench::Args{-expo(0, 8), expo(-4, -2), 0}; }},
  };
  const st::par qp = qpar();
  for (auto& r : regs) {
    const std::size_t n = 4096;
    std::vector<bench::Args> v(n);
    for (auto& a : v) a = r.draw();
    long cnt = 0;
    st::par qc = qp;
    int c = 0;
    qc.count = &c;
    for (const auto& a : v) {
      double L[NC];
      st::derivs<5>(a[0], a[1], qc, L);
      cnt += c;
      c = 0;
    }
    const auto t = bench::make_time_args(n, [&, i = std::size_t(0)]() mutable { return v[i++]; });
    std::printf("%s: %.1f nodes per call\n", r.name, cnt / double(n));
    timing<1>("quad", Cand{0, qp}, t);
    timing<5>("quad", Cand{0, qp}, t);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_student <reference dir> [acc|all|worst|time|prof|deep|point z nu] [candidate]\n");
    return 2;
  }
  const std::string refdir = argv[1];
  const std::string mode = argc > 2 ? argv[2] : "acc";
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  if (mode == "prof") {
    profile();
    return 0;
  }
  if (mode == "deep") {
    deep();
    return 0;
  }
  if (mode == "cnu") {  // c(nu) and its derivatives against lgamma and polygamma
    const double nu = std::atof(argv[3]);
    double c[6];
    st::cpoly(nu, 5, c);
    const double c0 = std::lgamma(0.5 * (nu + 1)) - std::lgamma(0.5 * nu) - 0.5 * std::log(nu * 3.14159265358979323846);
    std::printf("c %.17g lgamma %.17g\n", c[0], c0);
    for (int j = 1; j <= 5; ++j) {
      // psi^(j-1)((nu+1)/2)/2^j - psi^(j-1)(nu/2)/2^j - d^j log(nu)/2
      const double p = laplace::math::polygamma(j - 1, 0.5 * (nu + 1)) - laplace::math::polygamma(j - 1, 0.5 * nu);
      const double ref = p / std::pow(2.0, j) - 0.5 * ((j % 2 == 1) ? 1.0 : -1.0) * ibd::factorial(j - 1) / std::pow(nu, j);
      std::printf("c^(%d) %.17g direct %.17g\n", j, c[j] * ibd::factorial(j), ref);
    }
    return 0;
  }
  if (mode == "point") {
    const double z = std::atof(argv[3]), nu = std::atof(argv[4]);
    for (int m = 0; m < 2; ++m) {
      double L[NC];
      if (m == 0) st::derivs<5>(z, nu, qpar(), L);
      else st::derivs_beta<5>(z, nu, L);
      std::printf("%s: value %.17g |", m ? "beta" : "quad", m ? st::log_cdf_beta(z, nu) : st::log_cdf_quad(z, nu));
      for (int nn = 0; nn <= 5; ++nn)
        for (int j = 0; j <= nn; ++j)
          std::printf(" %.17g", L[st::ix(nn - j, j)] * ibd::factorial(nn - j) * ibd::factorial(j));
      std::printf("\n");
    }
    return 0;
  }
  const Refs R = load(refdir);
  const std::size_t n = std::size_t(1) << 13;
  std::mt19937_64 gen(20261109);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> nz(0.0, 1.0);
  const auto t = bench::make_time_args(n, [&] { return bench::Args{3.0 * nz(gen), std::exp2(-2.0 + 12.0 * u01(gen)), 0.0}; });
  if (filter.empty() || filter == "value") value(R, t, ".", mode);
  if (filter == "value") return 0;
  run_order<1>(R, mode, filter, t);
  run_order<2>(R, mode, filter, t);
  run_order<3>(R, mode, filter, t);
  run_order<4>(R, mode, filter, t);
  run_order<5>(R, mode, filter, t);
  return 0;
}
