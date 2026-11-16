// Accuracy and speed of weibull_distrib (include/laplace/distrib/weibull.hpp).
//
// Accuracy: the library's value and derivative orders of log f, log F and log(1 - F) against the mpmath
// references of tools/reference/distrib_weibull.py (host only), as the scaled ulp of
// tests/distrib/test_weibull.cpp (the error of a component times m^a sigma^b, divided by the ulp of the
// largest such component of the order at the same point and by the conditioning of h = sigma log(y/b)),
// the worst value per order with its arguments, and the worst error of the expected information
// relative to its natural scale.
//
// Speed: ns per call of the value, orders 1 and 6, the mixed derivative (2, 3), the cdf and log cdf with the
// derivatives of order 2 and 6 of log F and log(1 - F), the quantile, the information and its fourth
// derivative, on one host thread (Serial) and on the default execution space (OpenMP or CUDA).
//
//   distrib_weibull [reference directory]
#include "measure.hpp"

#include <laplace/distrib/weibull.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#define LAPLACE_REFERENCE_DIR "tests/data/reference"
#endif

namespace ld = laplace::distrib;

namespace {

struct wrow {
  double m = 0, s = 0, y = 0, hi = 0, lo = 0;
  int K = 0, b = 0;
};

std::vector<wrow> read_rows(const std::string& path) {
  std::vector<wrow> rows;
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return rows;
  }
  char line[512], a[64], b[64], c[64], hi[64], lo[64];
  int K, bb;
  if (!std::fgets(line, sizeof line, f)) return rows;
  while (std::fgets(line, sizeof line, f)) {
    if (std::sscanf(line, "%63[^,],%63[^,],%63[^,],%d,%d,%63[^,],%63s", a, b, c, &K, &bb, hi, lo) != 7) continue;
    wrow r;
    r.m = std::strtod(a, nullptr), r.s = std::strtod(b, nullptr), r.y = std::strtod(c, nullptr);
    r.K = K, r.b = bb, r.hi = std::strtod(hi, nullptr), r.lo = std::strtod(lo, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

double log_ratio(double y, double m) {
  const double q = y / m;
  return (q > 1e-250 && q < 1e250) ? std::log(q) : std::log(y) - std::log(m);
}

// the component b of order K of file (0 log f, 1 log F, 2 log(1 - F))
template <int K>
double component(const ld::weibull_distrib& d, int file, double y, int b) {
  if (file == 0) return d.logpdf_dk<K>(y)[b];
  if (file == 1) return d.log_cdf_dk<K>(y)[b];
  return d.log_ccdf_dk<K>(y)[b];
}

double library_value(const ld::weibull_distrib& d, int file, const wrow& r) {
  switch (r.K) {
    case 0: return file == 0 ? d.logpdf(r.y) : (file == 1 ? d.log_cdf(r.y) : d.log_ccdf(r.y));
    case 1: return component<1>(d, file, r.y, r.b);
    case 2: return component<2>(d, file, r.y, r.b);
    case 3: return component<3>(d, file, r.y, r.b);
    case 4: return component<4>(d, file, r.y, r.b);
    case 5: return component<5>(d, file, r.y, r.b);
    default: return component<6>(d, file, r.y, r.b);
  }
}

void accuracy(const std::string& dir) {
  const char* names[3] = {"weibull_logpdf.csv", "weibull_log_cdf.csv", "weibull_log_ccdf.csv"};
  const char* what[3] = {"log f", "log F", "log(1 - F)"};
  for (int file = 0; file < 3; ++file) {
    const auto rows = read_rows(dir + "/" + names[file]);
    std::map<std::tuple<double, double, double, int>, double> scale;
    const auto weight = [](const wrow& r) { return std::pow(r.m, r.K - r.b) * std::pow(r.s, r.b); };
    for (const auto& r : rows) {
      auto& s = scale[std::make_tuple(r.m, r.s, r.y, r.K)];
      s = std::fmax(s, std::fabs(r.hi) * weight(r));
    }
    double worst[7] = {}, wm[7] = {}, ws[7] = {}, wy[7] = {};
    int wb[7] = {};
    for (const auto& r : rows) {
      const ld::weibull_distrib d{r.m, r.s};
      const double v = library_value(d, file, r);
      const double err = std::fabs((v - r.hi) - r.lo) * weight(r);
      double cond = 1.0 + r.s * std::fabs(log_ratio(r.y, r.m));
      if (file == 1) cond *= 1.0 + std::exp(d.hazard_log(r.y));
      if (r.K == 1 && r.b == 0) cond *= std::fmax(1.0, r.s);
      const double e = err / ulp_of(scale[std::make_tuple(r.m, r.s, r.y, r.K)]) / cond;
      if (e > worst[r.K]) worst[r.K] = e, wm[r.K] = r.m, ws[r.K] = r.s, wy[r.K] = r.y, wb[r.K] = r.b;
    }
    std::printf("%s (%zu rows): worst scaled ulp per order, with the arguments (m, sigma, y) and the component b\n",
                what[file], rows.size());
    for (int k = 0; k <= 6; ++k)
      std::printf("  order %d: %9.3g  (%.6g, %.6g, %.6g)  b = %d\n", k, worst[k], wm[k], ws[k], wy[k], wb[k]);
  }
  // the expected information
  std::FILE* f = std::fopen((dir + "/weibull_info.csv").c_str(), "r");
  if (!f) return;
  char line[512];
  std::fgets(line, sizeof line, f);
  double worst[3] = {}, wsig[3] = {};
  while (std::fgets(line, sizeof line, f)) {
    double v[8];
    char* p = line;
    for (int i = 0; i < 8; ++i) {
      char* end;
      v[i] = std::strtod(p, &end);
      p = end + 1;
    }
    const ld::weibull_distrib d{v[0], v[1]};
    const auto e = d.expected_info();
    const double ref[3] = {v[2] + v[3], v[4] + v[5], v[6] + v[7]};
    const double sc[3] = {ref[0], std::sqrt(ref[0] * ref[2]), ref[2]};
    for (int i = 0; i < 3; ++i) {
      const double err = std::fabs(e[i] - ref[i]) / sc[i];
      if (err > worst[i]) worst[i] = err, wsig[i] = v[1];
    }
  }
  std::fclose(f);
  std::printf("expected information, relative to the natural scale: I_mm %.3g (sigma %.4g), I_m,sigma %.3g (sigma %.4g), "
              "I_sigma,sigma %.3g (sigma %.4g)\n",
              worst[0], wsig[0], worst[1], wsig[1], worst[2], wsig[2]);
}

// ---------------------------------------------------------------- speed

// the quantity by a code; the arguments are (y, m, sigma)
template <int WHAT>
struct weibull_f {
  KOKKOS_INLINE_FUNCTION double operator()(double y, double m, double sigma) const {
    const ld::weibull_distrib d{m, sigma};
    double s = 0.0;
    if constexpr (WHAT == 0) return d.logpdf(y);
    else if constexpr (WHAT == 1) {
      const auto t = d.logpdf_d1(y);
      return t[0] + t[1];
    } else if constexpr (WHAT == 2) {
      const auto t = d.logpdf_d6(y);
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 3) {
      const auto t = d.logpdf_dy_dk<2, 3>(y);
      for (int b = 0; b <= 3; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 4) return d.cdf(y) + d.log_cdf(y) + d.log_ccdf(y);
    else if constexpr (WHAT == 5) {
      const auto t = d.log_cdf_d2(y);
      return t[0] + t[1] + t[2];
    } else if constexpr (WHAT == 6) {
      const auto t = d.log_cdf_d6(y);
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 7) {
      const auto t = d.log_ccdf_d6(y);
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 8) return d.quantile(0.37) + d.quantile_from_log(-3.0);
    else if constexpr (WHAT == 9) {
      const auto t = d.expected_info();
      return t[0] + t[1] + t[2];
    } else {
      const auto t = d.expected_info_d4();
      for (int p = 0; p < 3; ++p)
        for (int b = 0; b <= 4; ++b) s += t.pair[p][b];
      return s;
    }
  }
};

template <class F>
void time_line(const char* what, F f, const bench::TimeArgs& t, int reps = 3) {
  std::printf("%-34s | ns/call Serial %9.1f, %s %9.2f\n", what, bench::time_ns<bench::HostOne>(t.host, f, reps),
              bench::Exec::name(), bench::time_ns<bench::Exec>(t.dev, f, reps));
}

void speed_lines(const char* name, double m, double sigma, int n) {
  std::mt19937_64 g(20261115);
  std::uniform_real_distribution<double> u(0.02, 0.98);
  const ld::weibull_distrib d{m, sigma};
  const auto t = bench::make_time_args(n, [&] { return bench::Args{d.quantile(u(g)), m, sigma}; });
  const char* q[11] = {"log f",      "log f d1",   "log f d6", "log f dy(2,3)", "cdf, log F, log(1-F)", "log F d2",
                       "log F d6",   "log(1-F) d6", "quantile x2", "info",        "info d4"};
  char what[96];
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[0]), what), weibull_f<0>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[1]), what), weibull_f<1>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[2]), what), weibull_f<2>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[3]), what), weibull_f<3>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[4]), what), weibull_f<4>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[5]), what), weibull_f<5>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[6]), what), weibull_f<6>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[7]), what), weibull_f<7>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[8]), what), weibull_f<8>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[9]), what), weibull_f<9>{}, t);
  time_line((std::snprintf(what, sizeof what, "%s %s", name, q[10]), what), weibull_f<10>{}, t);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    accuracy(argc > 1 && argv[1][0] != '-' ? std::string(argv[1]) : std::string(LAPLACE_REFERENCE_DIR));
    speed_lines("weibull(2, 1.5)", 2.0, 1.5, 1 << 10);
    speed_lines("weibull(1, 0.4)", 1.0, 0.4, 1 << 10);
    speed_lines("weibull(1, 8) polygamma", 1.0, 8.0, 1 << 10);
    speed_lines("weibull(50, 300) series", 50.0, 300.0, 1 << 10);
  }
  Kokkos::finalize();
  return 0;
}
