// Accuracy and speed of the von Mises block of Bessel functions
// (include/laplace/math/bessel_i.hpp): log I0 and its scaled form, the ratio
// A = I1/I0 and its derivatives to order six, the inverse K of A and its
// derivatives to order six. Candidates: Laplace's own (with a sweep of its
// switch points), Boost.Math's I0 and I1, and for log I0 Laplace's first
// candidate (series and asymptotic expansion) against the chosen one (Boost's
// rational approximations of I0 in log form).
//
// Usage: math_bessel <reference dir> <output dir> [name filter]
#include "measure.hpp"

#include <laplace/math/bessel_i.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/bessel.hpp>

#include <random>

namespace {

namespace db = laplace::math::own::detail_bessel;
using laplace::math::detail::boost_policy;

// ---------------------------------------------------------------- log I0
struct LogI0Chosen {  // the header's: Boost's rational approximations in log form
  bool scaled;
  KOKKOS_INLINE_FUNCTION double operator()(double k) const { return db::log_i0(k, scaled); }
};
struct LogI0Boost {  // log of Boost's I0 (overflows from k = 713)
  bool scaled;
  KOKKOS_INLINE_FUNCTION double operator()(double k) const {
    const double l = Kokkos::log(boost::math::cyl_bessel_i(0.0, k, boost_policy()));
    return scaled ? l - k : l;
  }
};
// Laplace's own first candidate: the series I0(k) = sum_m (k^2/4)^m / (m!)^2
// below k = 22, as log1p of its terms from m = 1; above, the asymptotic
// expansion e^-k sqrt(2 pi k) I0(k) ~ 1 + sum_n c_n k^-n,
// c_n = ((2n-1)!!)^2 / (n! 8^n), summed while its terms decrease.
struct LogI0Series {
  bool scaled;
  KOKKOS_INLINE_FUNCTION double operator()(double k) const {
    if (!(k >= 0.0)) return k * NAN;
    if (k < 22.0) {
      const double q = 0.25 * k * k;
      double t = 1.0, s = 0.0;
      for (int m = 1; m < 200; ++m) {
        t *= q / (double(m) * double(m));
        s += t;
        if (t <= 0x1p-56 * s) break;
      }
      const double l = Kokkos::log1p(s);
      return scaled ? l - k : l;
    }
    const double u = 1.0 / k;
    double t = 1.0, s = 0.0;
    for (int n = 1; n < 100; ++n) {
      const double tn = t * (double(2 * n - 1) * double(2 * n - 1)) / (8.0 * n) * u;
      if (tn >= t) break;
      t = tn;
      s += t;
      if (t <= 0x1p-56 * s) break;
    }
    const double l = -0.5 * (1.8378770664093454836 + laplace::math::log(k)) + Kokkos::log1p(s);
    return scaled ? l : k + l;
  }
};

// ---------------------------------------------------------------- the ratio and its derivatives
template <int M>
struct RatioOwn {
  db::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double k) const { return db::ratio_dm<M>(k, p); }
};
struct RatioBoost {  // I1/I0 (NaN from k = 713)
  KOKKOS_INLINE_FUNCTION double operator()(double k) const {
    return boost::math::cyl_bessel_i(1.0, k, boost_policy()) / boost::math::cyl_bessel_i(0.0, k, boost_policy());
  }
};

// ---------------------------------------------------------------- the inverse and its derivatives
struct InvOwn {
  db::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double rho) const { return db::ratio_inv(rho, p); }
};
template <int N>
struct InvDOwn {
  db::params p;
  KOKKOS_INLINE_FUNCTION double operator()(double k) const { return db::ratio_inv_dn<N>(k, p); }
};

db::params with(double small, double large, int extra) {
  db::params p;
  p.small = small, p.large = large, p.extra = extra;
  return p;
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_bessel <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 20;
  std::mt19937_64 gen(20261110);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  // kappa: half uniform on (0, 50), half log-uniform on (2^-4, 2^10); rho uniform
  const auto t_k = bench::make_time_args(n, [&] { return u01(gen) < 0.5 ? 50.0 * u01(gen) : std::exp2(-4.0 + 14.0 * u01(gen)); });
  const auto t_rho = bench::make_time_args(n, [&] { return u01(gen); });

  auto run = [&](const std::string& name, const std::string& impl, auto f, const bench::TimeArgs& t) {
    if (!filter.empty() && name.find(filter) == std::string::npos) return;
    bench::report(name.c_str(), impl.c_str(), bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t);
  };

  run("log_bessel_i0", "own", LogI0Chosen{false}, t_k);
  run("log_bessel_i0", "series", LogI0Series{false}, t_k);
  run("log_bessel_i0", "boost", LogI0Boost{false}, t_k);
  run("log_bessel_i0_scaled", "own", LogI0Chosen{true}, t_k);
  run("log_bessel_i0_scaled", "series", LogI0Series{true}, t_k);
  run("log_bessel_i0_scaled", "boost", LogI0Boost{true}, t_k);

  run("bessel_i_ratio", "boost", RatioBoost{}, t_k);
  // the switch points: (small, large, extra)
  const struct {
    const char* tag;
    db::params p;
  } sweep[] = {{"own", db::params{}},
               {"L40", with(0.25, 40.0, 20)},
               {"L60", with(0.25, 60.0, 20)},
               {"x30", with(0.25, 30.0, 30)},
               {"S05", with(0.5, 30.0, 20)}};
  for (const auto& s : sweep) {
    run("bessel_i_ratio", s.tag, RatioOwn<0>{s.p}, t_k);
    run("bessel_i_ratio_d1", s.tag, RatioOwn<1>{s.p}, t_k);
    run("bessel_i_ratio_d2", s.tag, RatioOwn<2>{s.p}, t_k);
    run("bessel_i_ratio_d3", s.tag, RatioOwn<3>{s.p}, t_k);
    run("bessel_i_ratio_d4", s.tag, RatioOwn<4>{s.p}, t_k);
    run("bessel_i_ratio_d5", s.tag, RatioOwn<5>{s.p}, t_k);
    run("bessel_i_ratio_d6", s.tag, RatioOwn<6>{s.p}, t_k);
    run("bessel_i_ratio_inv", s.tag, InvOwn{s.p}, t_rho);
    run("bessel_i_ratio_inv_d1", s.tag, InvDOwn<1>{s.p}, t_k);
    run("bessel_i_ratio_inv_d2", s.tag, InvDOwn<2>{s.p}, t_k);
    run("bessel_i_ratio_inv_d3", s.tag, InvDOwn<3>{s.p}, t_k);
    run("bessel_i_ratio_inv_d4", s.tag, InvDOwn<4>{s.p}, t_k);
    run("bessel_i_ratio_inv_d5", s.tag, InvDOwn<5>{s.p}, t_k);
    run("bessel_i_ratio_inv_d6", s.tag, InvDOwn<6>{s.p}, t_k);
  }
  return 0;
}
