// Accuracy and speed of Dawson's integral and of its derivatives to order six
// (include/laplace/math/dawson.hpp), F(x) = e^(-x^2) int_0^x e^(s^2) ds.
//
// Candidates, for the value and for every derivative order:
//   own-T     the header, with a sweep of the |x| = T at which the asymptotic
//             series takes over (5, 6, 6.5, 7, 8);
//   series    the scaled series alone, e^(-x^2) sum_m b_K(m) x^n/m!;
//   asym      the differentiated asymptotic series alone;
//   taylor    the alternating Taylor series of F, sum_n (-2)^n x^(2n+1)/(2n+1)!!,
//             differentiated term by term;
//   recur     the recurrence F^(1) = 1 - 2 x F, F^(k+1) = -2 (x F^(k) + k F^(k-1))
//             over the header's value, to measure the cancellation it costs.
// For the value only:
//   rybicki-h Rybicki's equally spaced sum, F(x) = (1/sqrt(pi)) sum_(n odd)
//             e^(-(x - n h)^2)/n, with the geometric recurrence that turns the
//             exponentials into multiplications;
//   cf-L      the continued fraction of the asymptotic series,
//             F = 1/(2 D), D = x - (1/2)/(x - 1/(x - (3/2)/(x - ...))), from L levels.
//
// Each candidate is reported over the whole reference grid and over the three
// regions on their own (|x| < 5/2, 5/2 to 8, from 8 up), timed on arguments
// drawn from the region, so that the cost of every route is read where it is
// used.
//
// Usage: math_dawson <reference dir> <output dir> [name filter]
#include "measure.hpp"

#include <laplace/math/dawson.hpp>

#include <random>

namespace {

namespace dd = laplace::math::detail_dawson;

// ---------------------------------------------------------------- the header, with the threshold swept
template <int K>
struct Own {
  double from;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    if (Kokkos::isnan(x)) return x;
    return Kokkos::fabs(x) < from ? dd::scaled_series<K>(x) : dd::asymptotic<K>(x);
  }
};

// ---------------------------------------------------------------- one route everywhere
template <int K>
struct Series {  // e^(x^2) overflows past |x| = 26.6, where the series cannot be used at all
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return dd::scaled_series<K>(x); }
};
template <int K>
struct Asym {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return dd::asymptotic<K>(x); }
};

// ---------------------------------------------------------------- the alternating Taylor series
// F^(K)(x) = sum_(n >= n0) ((-2)^n/(2n+1)!!) (2n+1)(2n)...(2n+2-K) x^(2n+1-K),
// n0 = K/2 (the first term that survives K differentiations). From n0 + 1 on,
// t_n/t_(n-1) = -2 x^2 (2n)/((2n - K)(2n + 1 - K)), whose factors no longer
// vanish, so the first term is built directly and the rest follow it.
template <int K>
struct Taylor {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    if (Kokkos::isnan(x)) return x;
    const double w = x * x;
    constexpr int n0 = K / 2;
    double t = 1.0;
    for (int i = 1; i <= n0; ++i) t *= -2.0 / double(2 * i + 1);    // (-2)^n0/(2 n0 + 1)!!
    for (int j = 0; j < K; ++j) t *= double(2 * n0 + 1 - j);        // the falling factorial
    if ((2 * n0 + 1 - K) == 1) t *= x;                              // the power of x is 0 or 1
    double sum = t, amax = Kokkos::fabs(t);
    for (int nn = n0 + 1; nn <= n0 + 400; ++nn) {
      t *= -2.0 * w * double(2 * nn) / (double(2 * nn - K) * double(2 * nn + 1 - K));
      if (!Kokkos::isfinite(t)) break;                              // the series has run away
      sum += t;
      const double at = Kokkos::fabs(t);
      if (at > amax) amax = at;
      if (at <= 0x1p-56 * amax) break;
    }
    return sum;
  }
};

// ---------------------------------------------------------------- Rybicki's equally spaced sum
// F(x) = (1/sqrt(pi)) sum_(n odd) e^(-(x - n h)^2)/n in the limit h -> 0, with
// the terms paired around the even multiple n0 h nearest to x, as in Rybicki
// (1989): with xp = |x| - n0 h, the term of n = n0 +- (2i - 1) is
// e^(-xp^2) e^(-((2i-1)h)^2) e^(+-2 xp (2i-1) h), and both exponentials follow
// a geometric recurrence, so the whole sum costs three exp calls.
struct Rybicki {
  double h;
  int terms;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    if (Kokkos::isnan(x)) return x;
    const double ax = Kokkos::fabs(x);
    const double n0 = 2.0 * Kokkos::floor(0.5 * ax / h + 0.5);
    const double xp = ax - n0 * h;
    const double e1 = Kokkos::exp(2.0 * xp * h), e2 = e1 * e1;  // e^(2 xp h) and its square
    const double r = Kokkos::exp(-8.0 * h * h);                 // c_(i+1)/c_i = r^i
    double c = Kokkos::exp(-h * h), g = r;                      // c_i = e^(-((2i-1) h)^2)
    double p = e1;                                              // e^(2 xp (2i-1) h)
    double d1 = n0 + 1.0, d2 = n0 - 1.0, sum = 0.0;
    for (int i = 1; i <= terms; ++i) {
      sum += c * (p / d1 + 1.0 / (d2 * p));
      c *= g;
      g *= r;
      p *= e2;
      d1 += 2.0;
      d2 -= 2.0;
      if (c == 0.0) break;
    }
    const double s = 0.56418958354775628695 * Kokkos::exp(-xp * xp) * sum;  // 1/sqrt(pi)
    return x < 0.0 ? -s : s;
  }
};

// ---------------------------------------------------------------- the continued fraction
struct Fraction {
  int levels;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    if (Kokkos::isnan(x)) return x;
    double d = x;
    for (int n = levels; n >= 1; --n) d = x - 0.5 * double(n) / d;
    return 1.0 / (2.0 * d);
  }
};

// ---------------------------------------------------------------- the derivative recurrence
template <int K>
struct Recur {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    const double f0 = laplace::math::dawson(x);
    if constexpr (K == 0) {
      return f0;
    } else {
      double f[K + 1];
      f[0] = f0;
      f[1] = 1.0 - 2.0 * x * f0;
      for (int k = 1; k < K; ++k) f[k + 1] = -2.0 * (x * f[k] + double(k) * f[k - 1]);
      return f[K];
    }
  }
};

// the rows of a reference file whose argument lies in [lo, hi) in absolute value
bench::Ref restrict_ref(const bench::Ref& r, double lo, double hi) {
  bench::Ref o;
  o.nargs = r.nargs;
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    const double a = std::fabs(r.x[i][0]);
    if (a >= lo && a < hi) {
      o.x.push_back(r.x[i]);
      o.hi.push_back(r.hi[i]);
      o.lo.push_back(r.lo[i]);
    }
  }
  return o;
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: math_dawson <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 18;
  std::mt19937_64 gen(20261118);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  // x: half uniform on (-12, 12), half log-uniform on (2^-10, 2^20) with a random sign
  const auto t_x = bench::make_time_args(n, [&] {
    if (u01(gen) < 0.5) return -12.0 + 24.0 * u01(gen);
    const double a = std::exp2(-10.0 + 30.0 * u01(gen));
    return u01(gen) < 0.5 ? -a : a;
  });
  const auto t_small = bench::make_time_args(n, [&] { return 2.5 * u01(gen); });
  const auto t_mid = bench::make_time_args(n, [&] { return 2.5 + 5.5 * u01(gen); });
  const auto t_large = bench::make_time_args(n, [&] { return std::exp2(3.0 + 17.0 * u01(gen)); });  // |x| >= 8

  const char* order[7] = {"dawson", "dawson_d1", "dawson_d2", "dawson_d3", "dawson_d4", "dawson_d5", "dawson_d6"};
  std::vector<bench::Ref> ref(7);
  for (int k = 0; k <= 6; ++k) ref[k] = bench::read_ref(refdir + "/" + std::string(order[k]) + ".csv");

  auto run = [&](const std::string& name, const std::string& impl, auto f, const bench::Ref& r,
                 const bench::TimeArgs& t) {
    if (!filter.empty() && (name + " " + impl).find(filter) == std::string::npos) return;
    bench::report(name.c_str(), impl.c_str(), r, f, outdir, t);
  };

  const double sweep[5] = {5.0, 6.0, 6.5, 7.0, 8.0};
  const char* tag[5] = {"own-5", "own-6", "own-6.5", "own-7", "own-8"};
  struct Region {
    const char* tag;
    double lo, hi;
    const bench::TimeArgs* t;
  };
  const Region region[3] = {{"small", 0.0, 2.5, &t_small}, {"mid", 2.5, 8.0, &t_mid}, {"large", 8.0, 1e308, &t_large}};

  auto all_orders = [&](auto&& per_order) {
    per_order(std::integral_constant<int, 0>{});
    per_order(std::integral_constant<int, 1>{});
    per_order(std::integral_constant<int, 2>{});
    per_order(std::integral_constant<int, 3>{});
    per_order(std::integral_constant<int, 4>{});
    per_order(std::integral_constant<int, 5>{});
    per_order(std::integral_constant<int, 6>{});
  };

  // the whole grid: the chosen route with its threshold swept, and the recurrence
  all_orders([&](auto kc) {
    constexpr int K = decltype(kc)::value;
    for (int s = 0; s < 5; ++s) run(order[K], tag[s], Own<K>{sweep[s]}, ref[K], t_x);
    run(order[K], "recur", Recur<K>{}, ref[K], t_x);
  });

  // each region on its own
  all_orders([&](auto kc) {
    constexpr int K = decltype(kc)::value;
    for (const auto& g : region) {
      const auto r = restrict_ref(ref[K], g.lo, g.hi);
      if (r.x.empty()) continue;
      const std::string name = std::string(order[K]) + "-" + g.tag;
      if (g.hi <= 26.0) run(name, "series", Series<K>{}, r, *g.t);
      run(name, "asym", Asym<K>{}, r, *g.t);
      if (g.hi <= 26.0) run(name, "taylor", Taylor<K>{}, r, *g.t);
      run(name, "recur", Recur<K>{}, r, *g.t);
    }
  });

  // the value only: Rybicki's sum and the continued fraction
  const double hs[4] = {0.4, 0.3, 0.22, 0.2};
  const int rterms[4] = {8, 12, 20, 24};
  const int levels[4] = {8, 16, 32, 64};
  for (const auto& g : region) {
    const auto r = restrict_ref(ref[0], g.lo, g.hi);
    for (int i = 0; i < 4; ++i) {
      char nm[32];
      std::snprintf(nm, sizeof nm, "rybicki-%.2f", hs[i]);
      run(std::string("dawson-") + g.tag, nm, Rybicki{hs[i], rterms[i]}, r, *g.t);
      std::snprintf(nm, sizeof nm, "cf-%d", levels[i]);
      run(std::string("dawson-") + g.tag, nm, Fraction{levels[i]}, r, *g.t);
    }
  }
  return 0;
}
