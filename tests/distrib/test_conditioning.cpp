// The derivatives of log F and log(1 - F) of the three gaussians at an argument that is
// exact in double (mu = 0, every scale 1, so z = y), against the mpmath references of
// tools/reference/distrib_conditioning.py. Measuring there separates two things that the
// grids of test_gaussian2.cpp and test_gaussian3.cpp see together: the error the route
// itself makes, and the error the route inherits because z = (y - mu)/sigma is already
// rounded. At large positive z the logarithmic derivative of L_1 = R(z) = phi(z)/Phi(z) is
// about -z, so half an ulp on z, that is z eps/2 in absolute terms, becomes a relative
// error of about z^2 eps/2 in the result: 85 ulp at z = 13 and 240 at z = 22, whatever
// route is used. This case pins the route's own error, which is what can be improved.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/gaussian3.hpp>
#include <laplace/distrib/gumbel.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

struct cond_row {
  int dist, upper, K, b;
  double y, hi, lo;
};

std::vector<cond_row> read_rows() {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_conditioning.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  std::vector<cond_row> rows;
  while (std::fgets(line, sizeof line, f)) {
    cond_row r;
    char ys[64], his[64], los[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63s", &r.dist, &r.upper, ys, &r.K, &r.b,
                    his, los) != 7)
      continue;
    r.y = std::strtod(ys, nullptr);
    r.hi = std::strtod(his, nullptr);
    r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

std::vector<double> route(const cond_row& r) {
  if (r.dist == 0) return cderiv(ld::gaussian1_distrib{0.0, 1.0}, r.y, r.K, r.upper != 0);
  if (r.dist == 1) return cderiv(ld::gaussian2_distrib{0.0, 1.0}, r.y, r.K, r.upper != 0);
  if (r.dist == 3) return cderiv(ld::gumbel_distrib{0.0, 1.0}, r.y, r.K, r.upper != 0);
  return cderiv(ld::gaussian3_distrib{0.0, 1.0}, r.y, r.K, r.upper != 0);
}

}  // namespace

TEST_CASE("distrib gaussians: log F derivatives at an exact argument") {
  const auto rows = read_rows();
  CHECK(rows.size() > 2400);
  // the error of each component in ulp of the largest component of its order at that y
  double worst[4][2] = {};
  double worst_y[4][2] = {};
  int worst_K[4][2] = {};
  for (const auto& r : rows) {
    const auto v = route(r);
    double scale = 0.0;
    for (const auto& q : rows)
      if (q.dist == r.dist && q.upper == r.upper && q.y == r.y && q.K == r.K)
        scale = std::fmax(scale, std::fabs(q.hi));
    if (scale == 0.0) continue;
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp = err / std::ldexp(1.0, std::ilogb(scale) - 52);
    if (ulp > worst[r.dist][r.upper]) {
      worst[r.dist][r.upper] = ulp;
      worst_y[r.dist][r.upper] = r.y;
      worst_K[r.dist][r.upper] = r.K;
    }
  }
  const char* name[4] = {"gaussian1 (sigma)", "gaussian2 (variance)", "gaussian3 (precision)",
                         "gumbel (sigma)"};
  for (int d = 0; d < 4; ++d)
    for (int u = 0; u < 2; ++u)
      if (std::getenv("LAPLACE_PRINT_ERRORS"))
        std::printf("%-22s %-10s exact z: max %.3g ulp of the order's scale at z = %g, K = %d\n",
                    name[d], u ? "log(1-F)" : "log F", worst[d][u], worst_y[d][u], worst_K[d][u]);
  // Measured 2026-11-14 on the gcc build, bounds about 1.5 times the maxima: gaussian1 38 ulp
  // at z = 4 and order 5, gaussian2 29 at z = 2 and order 6, gaussian3 3.08 at z = 8 and
  // order 6. gaussian2 read 117 at z = 3 and order 5 until the pieces of log_Phi_dK on
  // [-1, 4) were given a zero-aware form; its 29 is the chain's own cost, and it agrees with
  // the at most 28 ulp measured independently for that chain on exactly rounded inputs
  // (tools/reference/variance_chain.py). gaussian1 did not move because its worst point,
  // z = 4, is above that band. On the grids of test_gaussian2.cpp and test_gaussian3.cpp
  // nothing moved at all, 483 and 301 before and after, which is what the inherited error
  // predicts.
  // The three compute the same z-derivatives and differ only in the chain that
  // carries them to the parameters, so the gap between the columns is the chain's own cost.
  // It follows the sign of k in sigma = w^k: the recursion step is
  // -((k a + i) R_i + k z R_i'), whose two terms subtract when k > 0 (sigma, variance) and
  // add when k < 0 (precision), which is why the precision is the best conditioned of the
  // three and the variance the worst. On the grids of test_gaussian2.cpp and
  // test_gaussian3.cpp, where z = (y - mu)/sigma is not exact, these figures are buried under
  // the inherited z^2 eps/2.
  // The gumbel is here for the same reason: on its own grid the derivatives of log(1 - F)
  // reach 394 ulp, and with an exact argument they are 30.5, because log(1 - F) = -e^-u with
  // u = e^-z, so half an ulp on z becomes a relative |z| u eps/2 in the result, about 370 ulp
  // at the worst point of that grid (z = -5, u = 148). Its route needs no change; measured
  // 3.66 ulp for log F at z = 13 and order 5, 30.5 for log(1 - F) at z = -2 and order 6.
  const double bound[4][2] = {{57.0, 57.0}, {44.0, 44.0}, {4.6, 4.6}, {5.5, 46.0}};
  for (int d = 0; d < 4; ++d)
    for (int u = 0; u < 2; ++u) {
      INFO("dist " << d << " upper " << u << ": " << worst[d][u] << " ulp at z = " << worst_y[d][u]);
      CHECK(worst[d][u] <= bound[d][u]);
    }
}
