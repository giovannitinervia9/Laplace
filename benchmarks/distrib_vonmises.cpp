// Accuracy and speed of vonmises_distrib (include/laplace/distrib/vonmises.hpp).
//
// Accuracy: the value and every component of every order of the log-density against the mpmath references
// (tests/data/reference/vonmises_logpdf.csv), in ulp of the component and in ulp1, with the worst arguments,
// for s up to 4.5 and above it (where the conditioning of the map s -> kappa sets the error). A relative
// error is large near a zero of a component (sin z or cos z at a root, cos z = rho for a pure-s component),
// so the ulp column is read together with ulp1.
//
// Speed, ns per call on one host thread and on the default execution space: the value, each derivative
// order, the cdf and its logarithm (series and tail integral), the information, one draw by inversion, and
// the share of the order-six tensor that is the map s -> kappa (vonmises_kappa_from_s_upto). The derivatives
// of log F in the parameters are timed on the generic route (log_cdf_dk_by_integral) on a few points, which
// is the number that decides whether the family writes closed ones.
//
// Usage: distrib_vonmises <reference dir> [name filter]
#include "measure.hpp"

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/vonmises.hpp>

#include <cstdint>
#include <random>

namespace {

namespace ld = laplace::distrib;

// the component b of the order-K tensor (K = 0 the value) at (s, mu, y)
struct Component {
  int K, b;
  KOKKOS_INLINE_FUNCTION double operator()(double s, double mu, double y) const {
    const ld::vonmises_distrib d{mu, s};
    switch (K) {
      case 0: return d.logpdf(y);
      case 1: return d.logpdf_d1(y)[b];
      case 2: return d.logpdf_d2(y)[b];
      case 3: return d.logpdf_d3(y)[b];
      case 4: return d.logpdf_d4(y)[b];
      case 5: return d.logpdf_d5(y)[b];
      default: return d.logpdf_d6(y)[b];
    }
  }
};

// a whole tensor of order K, summed, for the timing
struct Tensor {
  int K;
  KOKKOS_INLINE_FUNCTION double operator()(double s, double mu, double y) const {
    const ld::vonmises_distrib d{mu, s};
    double acc = 0.0;
    switch (K) {
      case 0: return d.logpdf(y);
      case 1: { const auto t = d.logpdf_d1(y); for (int i = 0; i < 2; ++i) acc += t[i]; return acc; }
      case 2: { const auto t = d.logpdf_d2(y); for (int i = 0; i < 3; ++i) acc += t[i]; return acc; }
      case 3: { const auto t = d.logpdf_d3(y); for (int i = 0; i < 4; ++i) acc += t[i]; return acc; }
      case 4: { const auto t = d.logpdf_d4(y); for (int i = 0; i < 5; ++i) acc += t[i]; return acc; }
      case 5: { const auto t = d.logpdf_d5(y); for (int i = 0; i < 6; ++i) acc += t[i]; return acc; }
      default: { const auto t = d.logpdf_d6(y); for (int i = 0; i < 7; ++i) acc += t[i]; return acc; }
    }
  }
};

// only the map: kappa and its six derivatives
struct MapOnly {
  KOKKOS_INLINE_FUNCTION double operator()(double s, double, double) const {
    double kap[7];
    ld::vonmises_kappa_from_s_upto<6>(s, kap);
    double acc = 0.0;
    for (int i = 0; i <= 6; ++i) acc += kap[i];
    return acc;
  }
};

struct Cdf {
  int which;  // 0 F, 1 log F
  KOKKOS_INLINE_FUNCTION double operator()(double s, double mu, double y) const {
    const ld::vonmises_distrib d{mu, s};
    return which == 0 ? d.cdf(y) : d.log_cdf(y);
  }
};

struct Info {
  int K;
  KOKKOS_INLINE_FUNCTION double operator()(double s, double mu, double) const {
    const ld::vonmises_distrib d{mu, s};
    if (K == 0) {
      const auto e = d.expected_info();
      return e[0] + e[2];
    }
    const auto e = d.expected_info_d4();
    return e.pair[0][4] + e.pair[2][4];
  }
};

struct Draw {
  KOKKOS_INLINE_FUNCTION double operator()(double s, double mu, double y) const {
    const ld::vonmises_distrib d{mu, s};
    return ld::rng_by_inversion(d, std::uint64_t(7), std::uint64_t(Kokkos::fabs(y) * 1e9));
  }
};

// the derivatives of log F on the generic route
struct GenericLogCdf {
  int K;
  KOKKOS_INLINE_FUNCTION double operator()(double s, double mu, double y) const {
    const ld::vonmises_distrib d{mu, s};
    double acc = 0.0;
    if (K == 1) { const auto t = ld::log_cdf_dk_by_integral<1>(d, y); for (int i = 0; i < 2; ++i) acc += t[i]; }
    else if (K == 2) { const auto t = ld::log_cdf_dk_by_integral<2>(d, y); for (int i = 0; i < 3; ++i) acc += t[i]; }
    else { const auto t = ld::log_cdf_dk_by_integral<6>(d, y); for (int i = 0; i < 7; ++i) acc += t[i]; }
    return acc;
  }
};

// accuracy of one component over the rows of the file that carry it, split at s = 4.5
void accuracy_of(const std::vector<std::vector<double>>& rows, int K, int b) {
  for (int region = 0; region < 2; ++region) {
    bench::Ref r;
    r.nargs = 3;
    for (const auto& v : rows) {
      if (int(v[3]) != K || int(v[4]) != b || (v[0] > 4.5) != (region == 1)) continue;
      r.x.push_back({v[0], v[1], v[2]});
      r.hi.push_back(v[5]);
      r.lo.push_back(v[6]);
    }
    if (r.x.empty()) continue;
    const auto y = bench::eval_kernel(r.x, Component{K, b});
    const auto a = bench::accuracy(r, y);
    std::printf("order %d component %d, s %s 4.5 | %zu rows | max %9.4g ulp, %8.4g ulp1 | worst (s, mu, y) = %s | non-finite errors %zu\n",
                K, b, region ? "above" : "up to", r.x.size(), a.max_ulp, a.max_ulp1, bench::show(a.worst, 3).c_str(),
                a.n_bad);
  }
}

std::vector<std::vector<double>> read_rows(const std::string& path) {
  std::vector<std::vector<double>> rows;
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  char line[1024];
  if (!std::fgets(line, sizeof line, f)) return rows;
  while (std::fgets(line, sizeof line, f)) {
    std::vector<double> v;
    char* p = line;
    while (*p && *p != '\n') {
      char* end = nullptr;
      v.push_back(std::strtod(p, &end));
      if (*end != ',') break;
      p = end + 1;
    }
    rows.push_back(v);
  }
  std::fclose(f);
  return rows;
}

template <class F>
void time_line(const char* name, const bench::TimeArgs& t, F f, int reps_host, int reps_dev) {
  const double host = bench::time_ns<bench::HostOne>(t.host, f, reps_host);
  const double dev = bench::time_ns<bench::Exec>(t.dev, f, reps_dev);
  std::printf("%-34s | ns/call Serial %10.3f | %s %10.3f\n", name, host, bench::Exec::name(), dev);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: distrib_vonmises <reference dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1];
  const std::string filter = argc > 2 ? argv[2] : "";
  bench::print_spaces();
  const bool do_acc = filter.empty() || filter == "accuracy";
  const bool do_time = filter.empty() || filter == "time";

  if (do_acc) {
    const auto rows = read_rows(refdir + "/vonmises_logpdf.csv");  // s,mu,y,K,b,hi,lo,mag
    for (int K = 0; K <= 6; ++K)
      for (int b = 0; b <= K; ++b) accuracy_of(rows, K, b);
  }

  if (do_time) {
    const std::size_t n = std::size_t(1) << 16;
    std::mt19937_64 gen(20261115);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    // s log-uniform on 0.05 .. 3 (the working range), mu and y uniform on the circle
    const auto t = bench::make_time_args(n, [&] {
      return bench::Args{0.05 * std::exp(std::log(60.0) * u01(gen)), -3.0 + 6.0 * u01(gen), -3.0 + 6.0 * u01(gen)};
    });
    for (int K = 0; K <= 6; ++K) {
      char name[64];
      std::snprintf(name, sizeof name, "logpdf tensor, order %d", K);
      time_line(name, t, Tensor{K}, 3, 5);
    }
    time_line("map only (kappa, six derivatives)", t, MapOnly{}, 3, 5);
    time_line("cdf", t, Cdf{0}, 3, 5);
    time_line("log cdf", t, Cdf{1}, 3, 5);
    time_line("expected information", t, Info{0}, 3, 5);
    time_line("expected information, order 4", t, Info{1}, 3, 5);

    // the draw and the generic derivatives of log F are slow: a few thousand and a few dozen points
    const auto t_small = bench::make_time_args(std::size_t(2048), [&] {
      return bench::Args{0.05 * std::exp(std::log(60.0) * u01(gen)), -3.0 + 6.0 * u01(gen), -3.0 + 6.0 * u01(gen)};
    });
    time_line("draw by inversion", t_small, Draw{}, 1, 1);
    const auto t_tiny = bench::make_time_args(std::size_t(32), [&] {
      return bench::Args{0.2 + 2.0 * u01(gen), -2.0 + 4.0 * u01(gen), -3.0 + 6.0 * u01(gen)};
    });
    time_line("generic log F derivatives, order 1", t_tiny, GenericLogCdf{1}, 1, 1);
    time_line("generic log F derivatives, order 2", t_tiny, GenericLogCdf{2}, 1, 1);
    time_line("generic log F derivatives, order 6", t_tiny, GenericLogCdf{6}, 1, 1);
  }
  return 0;
}
