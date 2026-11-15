// Speed of pig_distrib (include/laplace/distrib/pig.hpp): the log-mass, its derivatives of orders 1
// and 6, the cdf and the derivatives of log F by the generic summation route, and the expected
// information and its second derivative, whose mean blocks are closed and whose shape block is a
// sum over the support. The half-integer Bessel sum is walked outwards from its largest term, so
// the lines at growing y show whether the cost follows the count or only the width of the weights.
// ns per call on one host thread (Serial) and on the default execution space. The accuracy is
// measured by tests/distrib/test_pig.cpp.
//
//   distrib_pig
#include "measure.hpp"

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/pig.hpp>

#include <cstdio>
#include <random>

namespace ld = laplace::distrib;

namespace {

template <int WHAT>
struct pig_f {
  KOKKOS_INLINE_FUNCTION double operator()(double y, double mu, double alpha) const {
    const ld::pig_distrib d{mu, alpha};
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

// the mass and its two derivative orders at a fixed count, to read the cost against y
void count_lines(double mu, double alpha, double y, int n) {
  const auto t = bench::make_time_args(n, [&] { return bench::Args{y, mu, alpha}; });
  char what[96];
  std::snprintf(what, sizeof what, "pig(%.3g, %.3g) log p at y = %.0g", mu, alpha, y);
  time_line(what, pig_f<0>{}, t);
  std::snprintf(what, sizeof what, "pig(%.3g, %.3g) log p d6 at y = %.0g", mu, alpha, y);
  time_line(what, pig_f<2>{}, t);
}

void family_lines(double mu, double alpha, int n) {
  std::mt19937_64 g(20261115);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const ld::pig_distrib d{mu, alpha};
  const double sd = d.scale();
  const auto t = bench::make_time_args(n, [&] {
    return bench::Args{std::floor(std::fmax(0.0, mu + sd * (4.0 * u(g) - 2.0))), mu, alpha};
  });
  char what[96], name[48];
  std::snprintf(name, sizeof name, "pig(%.3g, %.3g)", mu, alpha);
  const char* q[7] = {"log p", "log p d1", "log p d6", "log F (sum)", "log F d2 (sum)", "info (closed + sum)",
                      "info d2 (closed + sum)"};
  std::snprintf(what, sizeof what, "%s %s", name, q[0]);
  time_line(what, pig_f<0>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[1]);
  time_line(what, pig_f<1>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[2]);
  time_line(what, pig_f<2>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[3]);
  time_line(what, pig_f<3>{}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[4]);
  time_line(what, pig_f<4>{}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[5]);
  time_line(what, pig_f<5>{}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[6]);
  time_line(what, pig_f<6>{}, t, 1);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    family_lines(3.5, 2.0, 1 << 10);
    family_lines(1.0, 1e3, 1 << 8);  // the Poisson limit
    for (double y : {1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6}) count_lines(10.0, 1.0, y, 1 << 10);
  }
  Kokkos::finalize();
  return 0;
}
