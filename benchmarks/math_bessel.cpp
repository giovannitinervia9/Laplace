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


// ---------------------------------------------------------------- the ratios I_j/I_0 (von Mises cdf)
// R_j = I_j(k)/I_0(k), j = 1..m, for the series of the von Mises distribution
// function, m = max(20, ceil(8.5 sqrt k) + 10). Candidates (method):
//   0  statmodels7: r_j = I_j/I_(j-1) by the continued fraction backwards in
//      double from n0, R_j = r_1 ... r_j in double;
//   1  the same fraction, the products in double-double;
//   2  the fraction and the products in double-double;
//   3  Miller's backward recurrence y_(j-1) = (2j/k) y_j + y_(j+1), rescaled
//      against overflow, R_j = y_j / y_0.
// n0 = m + max(30, ceil(sqrt(40 m)), ceil(k)) (statmodels7's start) when
// start = 0, else m + start + ceil(k).
constexpr int max_terms = 512;
KOKKOS_INLINE_FUNCTION int cdf_terms(double k) {
  const int m = int(Kokkos::ceil(8.5 * Kokkos::sqrt(k))) + 10;
  return m > 20 ? m : 20;
}
KOKKOS_INLINE_FUNCTION void ratios(double k, int m, double* out, int method, int start, double csq = 0.0) {
  using laplace::math::detail::ddm::dd;
  namespace ddm = laplace::math::detail::ddm;
  int n0 = m + (start > 0 ? start : 30);
  if (start == 0) {
    const int s = int(Kokkos::ceil(Kokkos::sqrt(40.0 * m)));
    n0 = m + (s > 30 ? s : 30);
  }
  n0 += csq > 0.0 ? int(Kokkos::ceil(csq * Kokkos::sqrt(k))) : int(Kokkos::ceil(k));
  if (method == 0 || method == 1) {
    double r = 0.0;
    for (int j = n0; j >= 1; --j) {
      r = k / (2.0 * j + k * r);
      if (j <= m) out[j - 1] = r;
    }
    if (method == 0) {
      for (int j = 1; j < m; ++j) out[j] *= out[j - 1];
    } else {
      dd p(out[0]);
      for (int j = 1; j < m; ++j) {
        p = p * out[j];
        out[j] = p.hi;
      }
    }
  } else if (method == 2) {
    dd r(0.0), rr[max_terms];
    for (int j = n0; j >= 1; --j) {
      r = dd(k) / (k * r + 2.0 * j);
      if (j <= m) rr[j - 1] = r;
    }
    dd p = rr[0];
    out[0] = p.hi;
    for (int j = 1; j < m; ++j) {
      p = p * rr[j];
      out[j] = p.hi;
    }
  } else {
    double y1 = 0.0, y = 1e-280;  // y_(j+1), y_j
    for (int j = n0; j >= 1; --j) {
      const double ym = (2.0 * j) / k * y + y1;  // y_(j-1)
      y1 = y, y = ym;
      if (j - 1 >= 1 && j - 1 <= m) out[j - 2] = y;
      if (y > 1e250) {
        y *= 1e-250, y1 *= 1e-250;
        for (int i = j - 1; i <= m; ++i)
          if (i >= 1) out[i - 1] *= 1e-250;
      }
    }
    for (int i = 0; i < m; ++i) out[i] /= y;
  }
}
struct RatiosOwn {  // the header's (Miller, statmodels7's start)
  KOKKOS_INLINE_FUNCTION double operator()(double k, double jd) const {
    const int j = int(jd);
    int m = cdf_terms(k);
    if (j > m) m = j;
    if (m > max_terms) return k * NAN;
    double out[max_terms];
    laplace::math::bessel_i_ratios(k, m, out);
    return out[j - 1];
  }
};
struct RatiosF {
  int method, start;
  double csq = 0.0;  // > 0: start m + max(30, ceil(sqrt(40 m))) + ceil(csq sqrt(k))
  KOKKOS_INLINE_FUNCTION double operator()(double k, double jd) const {
    const int j = int(jd);
    int m = cdf_terms(k);
    if (j > m) m = j;
    if (m > max_terms) return k * NAN;
    double out[max_terms];
    ratios(k, m, out, method, start, csq);
    return out[j - 1];
  }
};

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

  {  // the ratios: time the whole array of the cdf series (k uniform on (0, 50))
    const auto t_kj = bench::make_time_args(n / 16, [&] { return bench::Args{50.0 * u01(gen), 1.0, 0.0}; });
    run("bessel_i_ratios", "own", RatiosOwn{}, t_kj);
    const char* tag[] = {"sm7", "ddprod", "ddall", "miller"};
    for (int meth = 0; meth < 4; ++meth) run("bessel_i_ratios", tag[meth], RatiosF{meth, 0}, t_kj);
    run("bessel_i_ratios", "mils5", RatiosF{3, 5}, t_kj);
    for (double c : {3.0, 5.0, 7.0, 9.0}) {
      char tag2[16];
      std::snprintf(tag2, sizeof tag2, "milq%g", c);
      run("bessel_i_ratios", tag2, RatiosF{3, 0, c}, t_kj);
    }
    // large concentrations: k uniform on (500, 1000)
    const auto t_big = bench::make_time_args(n / 64, [&] { return bench::Args{500.0 + 500.0 * u01(gen), 1.0, 0.0}; });
    run("bessel_i_ratios", "own-big", RatiosOwn{}, t_big);
    run("bessel_i_ratios", "milq7-big", RatiosF{3, 0, 7.0}, t_big);
  }
  return 0;
}
