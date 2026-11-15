// Speed of negbin1_distrib and negbin2_distrib (include/laplace/distrib/negbin1.hpp,
// negbin2.hpp): the log-mass, its derivatives of orders 1 and 6, the closed cdf through the
// incomplete beta, the derivatives of log F by the generic summation route, and the expected
// information and its second derivative by the generic sum over the support. ns per call on one
// host thread (Serial) and on the default execution space. The accuracy is measured by
// tests/distrib/test_negbin.cpp.
//
//   distrib_negbin
#include "measure.hpp"

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/negbin1.hpp>
#include <laplace/distrib/negbin2.hpp>

#include <cstdio>
#include <random>

namespace ld = laplace::distrib;

namespace {

// kind 0 negbin1, 1 negbin2; the quantity by a code
template <int FAM, int WHAT>
struct negbin_f {
  KOKKOS_INLINE_FUNCTION double operator()(double y, double mu, double theta) const {
    using D = std::conditional_t<FAM == 0, ld::negbin1_distrib, ld::negbin2_distrib>;
    const D d{mu, theta};
    if constexpr (WHAT == 0) return d.logpmf(y);
    else if constexpr (WHAT == 1) return d.logpmf_d1(y)[0] + d.logpmf_d1(y)[1];
    else if constexpr (WHAT == 2) {
      const auto t = d.logpmf_d6(y);
      double s = 0.0;
      for (int b = 0; b <= 6; ++b) s += t[b];
      return s;
    } else if constexpr (WHAT == 3) return d.log_cdf(y);
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

template <int FAM>
void family_lines(const char* name, double mu, double theta, int n) {
  std::mt19937_64 g(20261115);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double sd = FAM == 0 ? std::sqrt(mu * (1.0 + theta)) : std::sqrt(mu + mu * mu / theta);
  const auto t = bench::make_time_args(n, [&] {
    return bench::Args{std::floor(std::fmax(0.0, mu + sd * (4.0 * u(g) - 2.0))), mu, theta};
  });
  char what[96];
  const char* q[7] = {"log p", "log p d1", "log p d6", "log F (ibeta)", "log F d2 (sum)", "info (sum)",
                      "info d2 (sum)"};
  std::snprintf(what, sizeof what, "%s %s", name, q[0]);
  time_line(what, negbin_f<FAM, 0>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[1]);
  time_line(what, negbin_f<FAM, 1>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[2]);
  time_line(what, negbin_f<FAM, 2>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[3]);
  time_line(what, negbin_f<FAM, 3>{}, t);
  std::snprintf(what, sizeof what, "%s %s", name, q[4]);
  time_line(what, negbin_f<FAM, 4>{}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[5]);
  time_line(what, negbin_f<FAM, 5>{}, t, 1);
  std::snprintf(what, sizeof what, "%s %s", name, q[6]);
  time_line(what, negbin_f<FAM, 6>{}, t, 1);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    family_lines<1>("negbin2(3.5, 2)", 3.5, 2.0, 1 << 10);
    family_lines<1>("negbin2(40, 1e3) Poisson limit", 40.0, 1e3, 1 << 8);
    family_lines<0>("negbin1(3.5, 2)", 3.5, 2.0, 1 << 10);
    family_lines<0>("negbin1(2242, 0.028)", 2242.0, 0.028, 1 << 6);
  }
  Kokkos::finalize();
  return 0;
}
