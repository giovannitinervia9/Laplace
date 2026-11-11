// Accuracy and speed of log K_nu(x) and its derivatives in x to order 7
// (include/laplace/math/bessel_k.hpp). Candidates: Laplace's own with a sweep
// of its switch points, Boost.Math's cyl_bessel_k (value only, log of it),
// and the Riccati identity in double (to show its cancellation).
//
// Usage: math_bessel_k <reference dir> <output dir> [name filter]
#include "measure.hpp"

#include <laplace/math/bessel_k.hpp>

#include <boost/math/special_functions/bessel.hpp>

#include <random>

namespace {

namespace dk = laplace::math::own::detail_bessel_k;
using laplace::math::detail::boost_policy;

struct ValueOwn {
  dk::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const { return dk::log_k(nu, x, p); }
};
struct ValueBoost {  // NaN above nu = 1e5 (the recurrence is O(nu))
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const {
    if (nu > 1e5) return nu * NAN;
    return Kokkos::log(boost::math::cyl_bessel_k(nu, x, boost_policy()));
  }
};
struct ValueRec {  // Temme/Steed and the recurrence at every order (no Debye); NaN above 1e5
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const {
    if (nu > 1e5) return nu * NAN;
    double r;
    return dk::log_k_rec(Kokkos::fabs(nu), x, r);
  }
};

template <int N>
struct DerivOwn {
  dk::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const { return dk::derivative<N>(nu, x, p); }
};

// the Riccati identity in double from g in double, everywhere
template <int N>
struct DerivRiccatiDouble {  // NaN above nu = 1e5
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const {
    if (nu > 1e5) return nu * NAN;
    double r;
    dk::log_k_rec(nu, x, r);
    double d[N];
    d[0] = nu / x - r;
    double inv[N], inv2[N];
    double f = 1.0, f2 = 1.0, p = 1.0 / x, p2 = p * p;
    for (int m = 0; m < N; ++m) {
      f2 *= m + 1;
      inv[m] = (m % 2 ? -f : f) * p;
      inv2[m] = (m % 2 ? -f2 : f2) * p2;
      p /= x, p2 /= x, f *= m + 1;
    }
    for (int n = 0; n < N - 1; ++n) {
      double s = nu * nu * inv2[n], c = 1.0;
      for (int j = 0; j <= n; ++j) {
        s -= c * d[j] * (inv[n - j] + d[n - j]);
        c = c * (n - j) / (j + 1);
      }
      d[n + 1] = n == 0 ? 1.0 + s : s;
    }
    return d[N - 1];
  }
};

dk::params with(double nu_debye, double x_hankel, double x_cf, bool wallis = true) {
  dk::params p;
  p.nu_debye = nu_debye, p.x_hankel = x_hankel, p.x_cf = x_cf, p.wallis = wallis;
  return p;
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_bessel_k <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 18;
  std::mt19937_64 gen(20261111);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  // timing: nu in {0..4} or y - 1/2 with y up to 2^12, x log-uniform on (2^-4, 2^8)
  const auto t_nx = bench::make_time_args(n, [&] {
    const double nu = u01(gen) < 0.5 ? std::floor(5.0 * u01(gen)) : std::floor(std::exp2(12.0 * u01(gen))) - 0.5;
    return bench::Args{nu, std::exp2(-4.0 + 12.0 * u01(gen)), 0.0};
  });

  auto run = [&](const std::string& name, const std::string& impl, auto f) {
    if (!filter.empty() && (name + " " + impl).find(filter) == std::string::npos) return;
    bench::report(name.c_str(), impl.c_str(), bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t_nx);
  };

  run("log_bessel_k", "own", ValueOwn{});
  run("log_bessel_k", "rec", ValueRec{});
  run("log_bessel_k", "boost", ValueBoost{});
  run("log_bessel_k", "D10", ValueOwn{with(10.0, 200.0, 1.0)});
  run("log_bessel_k", "D30", ValueOwn{with(30.0, 200.0, 1.0)});

  const struct {
    const char* tag;
    dk::params p;
  } sweep[] = {{"own", dk::params{}},          {"D15", with(15.0, 200.0, 1.0)}, {"D25", with(25.0, 200.0, 1.0)},
               {"H100", with(20.0, 100.0, 1.0)}, {"H400", with(20.0, 400.0, 1.0)}, {"C05", with(20.0, 200.0, 0.5)},
               {"C2", with(20.0, 200.0, 2.0)},   {"steed", with(20.0, 200.0, 1.0, false)}};
  static const char* dname[] = {"", "log_bessel_k_dx1", "log_bessel_k_dx2", "log_bessel_k_dx3", "log_bessel_k_dx4",
                                "log_bessel_k_dx5", "log_bessel_k_dx6", "log_bessel_k_dx7"};
  for (const auto& s : sweep) {
    run(dname[1], s.tag, DerivOwn<1>{s.p});
    run(dname[2], s.tag, DerivOwn<2>{s.p});
    run(dname[3], s.tag, DerivOwn<3>{s.p});
    run(dname[4], s.tag, DerivOwn<4>{s.p});
    run(dname[5], s.tag, DerivOwn<5>{s.p});
    run(dname[6], s.tag, DerivOwn<6>{s.p});
    run(dname[7], s.tag, DerivOwn<7>{s.p});
  }
  // the cost by regime, on order 4: small orders 0..4 with x in (1, 16) (the
  // continued fraction in double-double), with x in (2^-8, 1) (Temme), x in
  // (200, 2^20) (asymptotic in 1/x), and orders y - 1/2 from 20 to 2^12 (Debye)
  {
    auto tr = [&](double lo_nu, double hi_nu, bool half, double lx0, double lx1) {
      return bench::make_time_args(n / 4, [&] {
        double nu = lo_nu + (hi_nu - lo_nu) * u01(gen);
        nu = half ? std::floor(nu) + 0.5 : std::floor(nu);
        return bench::Args{nu, std::exp2(lx0 + (lx1 - lx0) * u01(gen)), 0.0};
      });
    };
    const auto t_cf = tr(0.0, 5.0, false, 0.0, 4.0), t_te = tr(0.0, 5.0, false, -8.0, 0.0);
    const auto t_ha = tr(0.0, 5.0, false, 7.7, 20.0), t_de = tr(20.0, 4096.0, true, -4.0, 12.0);
    if (filter.empty() || filter == "time") {
      const auto r4 = bench::read_ref(refdir + "/log_bessel_k_dx4.csv");
      bench::report("dx4", "cf-dd", r4, DerivOwn<4>{}, outdir, t_cf);
      bench::report("dx4", "cf-steed", r4, DerivOwn<4>{with(20.0, 200.0, 1.0, false)}, outdir, t_cf);
      bench::report("dx4", "temme", r4, DerivOwn<4>{}, outdir, t_te);
      bench::report("dx4", "hankel", r4, DerivOwn<4>{}, outdir, t_ha);
      bench::report("dx4", "debye", r4, DerivOwn<4>{}, outdir, t_de);
      const auto r0 = bench::read_ref(refdir + "/log_bessel_k.csv");
      bench::report("value", "small", r0, ValueOwn{}, outdir, t_cf);
      bench::report("value", "debye", r0, ValueOwn{}, outdir, t_de);
    }
  }
  run(dname[2], "ricd", DerivRiccatiDouble<2>{});
  run(dname[4], "ricd", DerivRiccatiDouble<4>{});
  run(dname[7], "ricd", DerivRiccatiDouble<7>{});
  return 0;
}
