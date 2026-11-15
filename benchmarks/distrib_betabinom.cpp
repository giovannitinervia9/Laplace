// Speed of betabinom_distrib (include/laplace/distrib/betabinom.hpp): the log-mass, its
// derivatives of orders 1 and 6, the cdf and the derivatives of log F by the generic summation
// route, and the expected information and its second derivative by the generic sum over the
// support. The family has two routes, the sums over the integer shifts and the form of bounded cost
// through Stirling's error, and the members take whichever of them the size calls for, so the
// lines of the first block are read against n and the last block times the two routes side by side
// at the same points. ns per call on one host thread (Serial) and on the default execution space.
// The accuracy is measured by tests/distrib/test_betabinom.cpp.
//
//   distrib_betabinom
#include "measure.hpp"

#include <laplace/distrib/betabinom.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>

#include <cstdio>
#include <random>

namespace ld = laplace::distrib;

namespace {

// the quantity by a code; the size is a field of the functor, since the timing arguments carry
// three doubles and the size is fixed along a line
template <int WHAT>
struct betabinom_f {
  double size;
  KOKKOS_INLINE_FUNCTION double operator()(double y, double mu, double sigma) const {
    const ld::betabinom_distrib d{mu, sigma, size};
    if constexpr (WHAT == 0) return d.logpmf(y);
    else if constexpr (WHAT == 1) {
      // one call, both components summed: two calls each using one component let the compiler drop
      // the other, and the two components of this family cost very different amounts
      const auto t = d.logpmf_d1(y);
      return t[0] + t[1];
    }
    else if constexpr (WHAT == 2) {
      const auto t = d.logpmf_d6(y);
      double s = 0.0;
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 3) return ld::distrib_log_cdf(d, y);
    else if constexpr (WHAT == 4) {
      const auto t = ld::distrib_log_cdf_dk<2>(d, y);
      return t[0] + t[1] + t[2];
    } else if constexpr (WHAT == 5) {
      const auto t = ld::distrib_expected_info<0>(d);
      return t[0] + t[1] + t[2];
    } else {
      const auto t = ld::distrib_expected_info<2>(d);
      double s = 0.0;
      for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 3; ++i) s += t.pair[p][i];
      return s;
    }
  }
};

template <class F>
void time_line(const char* what, F f, const bench::TimeArgs& t, int reps = 3) {
  std::printf("%-44s | ns/call Serial %11.1f, %s %10.2f\n", what, bench::time_ns<bench::HostOne>(t.host, f, reps),
              bench::Exec::name(), bench::time_ns<bench::Exec>(t.dev, f, reps));
}

void size_lines(double mu, double sigma, double size, int n, bool heavy) {
  std::mt19937_64 g(20261115);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const auto t = bench::make_time_args(n, [&] {
    return bench::Args{std::floor(u(g) * (size + 1.0)), mu, sigma};
  });
  char what[96], name[48];
  std::snprintf(name, sizeof name, "betabinom(%.3g, %.3g; n = %.0f)", mu, sigma, size);
  const char* q[7] = {"log p", "log p d1", "log p d6", "log F (sum)", "log F d2 (sum)", "info (sum)",
                      "info d2 (sum)"};
  std::snprintf(what, sizeof what, "%s %s", name, q[0]);
  time_line(what, betabinom_f<0>{size}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[1]);
  time_line(what, betabinom_f<1>{size}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[2]);
  time_line(what, betabinom_f<2>{size}, t);
  if (!heavy) return;  // the cdf and the information are sums over the support of sums over it
  std::snprintf(what, sizeof what, "%s %s", name, q[3]);
  time_line(what, betabinom_f<3>{size}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[4]);
  time_line(what, betabinom_f<4>{size}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[5]);
  time_line(what, betabinom_f<5>{size}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[6]);
  time_line(what, betabinom_f<6>{size}, t, 1);
}

// the two routes at the same point, to read where their costs cross
template <int WHAT, bool STIRLING>
struct route_f {
  double size;
  KOKKOS_INLINE_FUNCTION double operator()(double y, double mu, double sigma) const {
    namespace db = ld::detail_betabinom;
    if constexpr (WHAT == 0)
      return STIRLING ? db::logpmf_stirling(mu, sigma, size, y) : db::logpmf_sums(mu, sigma, size, y);
    else {
      double t[WHAT + 1];
      if constexpr (STIRLING) db::logpmf_dk_stirling<WHAT>(mu, sigma, size, y, t);
      else db::logpmf_dk_sums<WHAT>(mu, sigma, size, y, t);
      double s = 0.0;
      for (int b = 0; b <= WHAT; ++b) s += t[b];
      return s;
    }
  }
};

void route_lines(double mu, double sigma, double size, int n) {
  const auto t = bench::make_time_args(n, [&] { return bench::Args{std::floor(mu * size) + 1.0, mu, sigma}; });
  char what[96];
  std::snprintf(what, sizeof what, "betabinom n = %.0f log p sums", size);
  time_line(what, route_f<0, false>{size}, t);
  std::snprintf(what, sizeof what, "betabinom n = %.0f log p stirling", size);
  time_line(what, route_f<0, true>{size}, t);
  std::snprintf(what, sizeof what, "betabinom n = %.0f d1 sums", size);
  time_line(what, route_f<1, false>{size}, t);
  std::snprintf(what, sizeof what, "betabinom n = %.0f d1 stirling", size);
  time_line(what, route_f<1, true>{size}, t);
  std::snprintf(what, sizeof what, "betabinom n = %.0f d6 sums", size);
  time_line(what, route_f<6, false>{size}, t);
  std::snprintf(what, sizeof what, "betabinom n = %.0f d6 stirling", size);
  time_line(what, route_f<6, true>{size}, t);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    size_lines(0.4, 0.5, 10.0, 1 << 10, true);
    size_lines(0.4, 0.5, 100.0, 1 << 8, true);
    size_lines(0.4, 0.05, 1000.0, 1 << 6, false);
    size_lines(0.4, 0.01, 10000.0, 1 << 4, false);
    // the two routes side by side, inside the region of the second (10 sigma <= min(mu, 1 - mu))
    for (double size : {16.0, 160.0, 1000.0, 10000.0}) route_lines(0.4, 0.002, size, 1 << 6);
  }
  Kokkos::finalize();
  return 0;
}
