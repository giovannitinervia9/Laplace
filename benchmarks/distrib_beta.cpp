// Accuracy and speed of beta_distrib (include/laplace/distrib/beta.hpp).
//
// Accuracy: the log-density and its derivatives of orders 1 to 6 against tests/data/reference/beta_logpdf.csv
// (tools/reference/distrib_beta.py, mpmath), on the host. Per order, the worst error in ulp of the largest
// component of the same (mu, phi, y, order) and, for the pure phi components, in ulp of the component
// itself, by region of phi: this is the large-phi cancellation, where the polygamma terms of those
// components cancel to one order of phi and the error grows like phi. The worst arguments are printed.
//
// Speed: ns per call on one host thread (Serial) and on the default execution space, for the value, the
// derivatives of orders 1, 2 and 6, the information and its derivatives, the cdf through the incomplete
// beta, the closed derivatives of log F (orders 1, 2 and 6) and the same through the generic quadrature
// route of generic_cdf.hpp (the member hidden), which is the argument for the closed route.
//
//   distrib_beta [reference directory]
#include "measure.hpp"

#include <laplace/distrib/beta.hpp>
#include <laplace/distrib/generic_cdf.hpp>

#include "../tests/distrib/generic_hidden.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

double ulp_at(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

template <int K>
std::vector<double> logpdf_tensor(const ld::beta_distrib& d, double y) {
  const auto t = d.logpdf_dk<K>(y);
  return std::vector<double>(t.v, t.v + ld::sym_tensor<2, K>::size);
}

std::vector<double> tensor_of(const ld::beta_distrib& d, double y, int K) {
  switch (K) {
    case 2: return logpdf_tensor<2>(d, y);
    case 3: return logpdf_tensor<3>(d, y);
    case 4: return logpdf_tensor<4>(d, y);
    case 5: return logpdf_tensor<5>(d, y);
    default: return logpdf_tensor<6>(d, y);
  }
}

void accuracy(const std::string& dir) {
  std::ifstream in(dir + "/beta_logpdf.csv");
  if (!in) {
    std::printf("cannot open %s/beta_logpdf.csv\n", dir.c_str());
    return;
  }
  struct row {
    double y, mu, phi, hi, lo;
    int K, b;
  };
  std::vector<row> rows;
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    row r;
    char ys[64], mus[64], ps[64], his[64], los[64];
    if (std::sscanf(line.c_str(), "%63[^,],%63[^,],%63[^,],%d,%d,%63[^,],%63s", ys, mus, ps, &r.K, &r.b, his, los) != 7)
      continue;
    r.y = std::strtod(ys, nullptr), r.mu = std::strtod(mus, nullptr), r.phi = std::strtod(ps, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    if (r.K >= 2) rows.push_back(r);
  }
  using key = std::tuple<double, double, double, int>;
  std::map<key, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[key(r.mu, r.phi, r.y, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  // buckets of phi for the pure phi components: <= 1e2, <= 1e4, <= 1e6, above
  const char* name[4] = {"phi <= 1e2", "1e2 < phi <= 1e4", "1e4 < phi <= 1e6", "phi > 1e6"};
  double scaled[7] = {}, own[7][4] = {};
  double at_scaled[7][3] = {}, at_own[7][4][3] = {};
  for (const auto& r : rows) {
    const ld::beta_distrib d{r.mu, r.phi};
    const auto t = tensor_of(d, r.y, r.K);
    const double err = std::fabs((t[r.b] - r.hi) - r.lo);
    const double u = err / ulp_at(scale[key(r.mu, r.phi, r.y, r.K)]);
    if (u > scaled[r.K]) {
      scaled[r.K] = u;
      at_scaled[r.K][0] = r.mu, at_scaled[r.K][1] = r.phi, at_scaled[r.K][2] = r.y;
    }
    if (r.b == r.K && r.hi != 0.0) {
      const int bk = r.phi <= 1e2 ? 0 : r.phi <= 1e4 ? 1 : r.phi <= 1e6 ? 2 : 3;
      const double o = err / ulp_at(r.hi);
      if (o > own[r.K][bk]) {
        own[r.K][bk] = o;
        at_own[r.K][bk][0] = r.mu, at_own[r.K][bk][1] = r.phi, at_own[r.K][bk][2] = r.y;
      }
    }
  }
  std::printf("accuracy of the order-K derivatives of log f against mpmath (%zu rows of order 2 to 6)\n", rows.size());
  for (int K = 2; K <= 6; ++K) {
    std::printf("  K = %d: worst %.4g ulp of the order's largest component at (mu %.6g, phi %.6g, y %.6g)\n", K,
                scaled[K], at_scaled[K][0], at_scaled[K][1], at_scaled[K][2]);
    for (int bk = 0; bk < 4; ++bk)
      std::printf("      pure phi component, %-17s: worst %.4g ulp of itself at (mu %.6g, phi %.6g, y %.6g)\n", name[bk],
                  own[K][bk], at_own[K][bk][0], at_own[K][bk][1], at_own[K][bk][2]);
  }
}

// the quantity to time, by a code
template <int WHAT>
struct beta_f {
  KOKKOS_INLINE_FUNCTION double operator()(double y, double mu, double phi) const {
    const ld::beta_distrib d{mu, phi};
    if constexpr (WHAT == 0) return d.logpdf(y);
    else if constexpr (WHAT == 1) return d.logpdf_d1(y)[0] + d.logpdf_d1(y)[1];
    else if constexpr (WHAT == 2) return d.logpdf_d2(y)[0] + d.logpdf_d2(y)[1] + d.logpdf_d2(y)[2];
    else if constexpr (WHAT == 3) {
      const auto t = d.logpdf_d6(y);
      double s = 0.0;
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 4) {
      const auto t = d.expected_info();
      return t[0] + t[1] + t[2];
    } else if constexpr (WHAT == 5) {
      const auto t = d.expected_info_d2();
      double s = 0.0;
      for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 3; ++i) s += t.pair[p][i];
      return s;
    } else if constexpr (WHAT == 6) {
      const auto t = d.expected_info_d4();
      double s = 0.0;
      for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 5; ++i) s += t.pair[p][i];
      return s;
    } else if constexpr (WHAT == 7) return d.log_cdf(y);
    else if constexpr (WHAT == 8) return d.log_cdf_d1(y)[0] + d.log_cdf_d1(y)[1];
    else if constexpr (WHAT == 9) {
      const auto t = d.log_cdf_d2(y);
      return t[0] + t[1] + t[2];
    } else if constexpr (WHAT == 10) {
      const auto t = d.log_cdf_d6(y);
      double s = 0.0;
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 11) return d.quantile(y);
    else if constexpr (WHAT == 12) {
      const auto t = ld::log_cdf_dk_by_integral<2>(laplace_test::hide(d), y);
      return t[0] + t[1] + t[2];
    } else {
      const auto t = ld::log_cdf_dk_by_integral<6>(laplace_test::hide(d), y);
      double s = 0.0;
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    }
  }
};

template <class F>
void time_line(const char* what, F f, const bench::TimeArgs& t, int reps = 3) {
  std::printf("%-46s | ns/call Serial %12.1f, %s %11.2f\n", what, bench::time_ns<bench::HostOne>(t.host, f, reps),
              bench::Exec::name(), bench::time_ns<bench::Exec>(t.dev, f, reps));
}

void speed_lines(const char* name, double mu, double phi, int n, int n_generic) {
  std::mt19937_64 g(20261115);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double sd = std::sqrt(mu * (1.0 - mu) / (phi + 1.0));
  auto draw_y = [&] {
    const double y = mu + sd * (4.0 * u(g) - 2.0);
    return y < 0.001 ? 0.001 : (y > 0.999 ? 0.999 : y);
  };
  const auto t = bench::make_time_args(static_cast<std::size_t>(n), [&] { return bench::Args{draw_y(), mu, phi}; });
  const auto tg = bench::make_time_args(static_cast<std::size_t>(n_generic), [&] { return bench::Args{draw_y(), mu, phi}; });
  const auto tq = bench::make_time_args(static_cast<std::size_t>(n), [&] { return bench::Args{0.02 + 0.96 * u(g), mu, phi}; });
  char what[96];
  auto line = [&](const char* q) { std::snprintf(what, sizeof what, "%s %s", name, q); return what; };
  time_line(line("log f"), beta_f<0>{}, t);
  time_line(line("log f d1"), beta_f<1>{}, t);
  time_line(line("log f d2"), beta_f<2>{}, t);
  time_line(line("log f d6"), beta_f<3>{}, t);
  time_line(line("info"), beta_f<4>{}, t);
  time_line(line("info d2"), beta_f<5>{}, t);
  time_line(line("info d4"), beta_f<6>{}, t);
  time_line(line("log F (ibeta)"), beta_f<7>{}, t);
  time_line(line("log F d1 (closed)"), beta_f<8>{}, t);
  time_line(line("log F d2 (closed)"), beta_f<9>{}, t);
  time_line(line("log F d6 (closed)"), beta_f<10>{}, t);
  time_line(line("quantile (from p)"), beta_f<11>{}, tq);
  time_line(line("log F d2 (generic quadrature)"), beta_f<12>{}, tg, 1);
  time_line(line("log F d6 (generic quadrature)"), beta_f<13>{}, tg, 1);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    accuracy(argc > 1 && argv[1][0] != '-' ? std::string(argv[1]) : std::string(LAPLACE_REFERENCE_DIR));
    speed_lines("beta(0.3, 6)", 0.3, 6.0, 1 << 10, 1 << 4);
    speed_lines("beta(0.5, 40)", 0.5, 40.0, 1 << 10, 1 << 4);
    speed_lines("beta(0.1, 2000)", 0.1, 2000.0, 1 << 10, 1 << 3);
  }
  Kokkos::finalize();
  return 0;
}
