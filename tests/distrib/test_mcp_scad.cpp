// The test of mcp_distrib and scad_distrib (include/laplace/distrib/mcp.hpp, scad.hpp),
// the properized MCP and SCAD: descriptors and kinks, the inner region against the
// original mcp_penalty and scad_penalty (an exact twin, the properization changing
// nothing there), the conventions at the kink u = 0 and at the knots, every derivative
// order against differences, the mixed (y, theta) derivatives, the closed cumulative
// functions, the normal and the Laplace limits, the mass by quadrature with and without
// the knots as cuts, the generic quantile and expected information, the references of
// tools/reference/distrib_mcp_scad.py, and a kernel launch per order.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/mcp.hpp>
#include <laplace/distrib/scad.hpp>
#include <laplace/math/quadrature.hpp>
#include <laplace/penalty/nonconvex_penalty.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"

namespace laplace_distrib_check {
// the shared checks build a distribution from a parameter vector, two parameters at most by default
template <>
ld::mcp_distrib make<ld::mcp_distrib>(const double* t) {
  return ld::mcp_distrib{t[0], t[1], t[2], t[3]};
}
template <>
ld::scad_distrib make<ld::scad_distrib>(const double* t) {
  return ld::scad_distrib{t[0], t[1], t[2], t[3]};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

using mcp = ld::mcp_distrib;
using scad = ld::scad_distrib;
namespace lp = laplace::penalty;

// ---------------------------------------------------------------- the four-parameter y-derivatives

#define Y4(J, K) \
  case J * 10 + K: return as_vec<4, K>(d.template logpdf_dy_dk<J, K>(y))
template <class D>
std::vector<double> yderiv4(const D& d, double y, int J, int K) {
  switch (J * 10 + K) {
    Y4(0, 0); Y4(0, 1); Y4(0, 2); Y4(0, 3); Y4(0, 4); Y4(0, 5); Y4(0, 6);
    Y4(1, 0); Y4(1, 1); Y4(1, 2); Y4(1, 3); Y4(1, 4); Y4(1, 5);
    Y4(2, 0); Y4(2, 1); Y4(2, 2); Y4(2, 3); Y4(2, 4);
    Y4(3, 0); Y4(3, 1); Y4(3, 2); Y4(3, 3);
    Y4(4, 0); Y4(4, 1); Y4(4, 2);
    Y4(5, 0); Y4(5, 1);
    Y4(6, 0);
    default: return {};
  }
}
#undef Y4

// every (J, K): against the difference in y of (J - 1, K) and in each parameter of (J, K - 1)
template <class D>
void check_y_derivs4(const std::vector<double>& th, double y, double hy, double rel) {
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yderiv4(make<D>(th.data()), y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yderiv4(make<D>(th.data()), y + s * hy, J - 1, K));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        INFO("dy: J = " << J << ", K = " << K << ", component " << i << ", y = " << y);
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12 + stencil_floor(v, i, hy));
      }
      if (K == 0) continue;
      for (const auto& idx : sorted_indices(4, K)) {
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const int j = idx.back();
        const double h = 1e-3 * std::fmax(1e-3, std::fabs(th[j]));
        const auto fd = diff5([&](const double* t) { return yderiv4(make<D>(t), y, J, K - 1); }, th, j, h);
        std::vector<std::vector<double>> w;
        for (double s : {-2.0, -1.0, 1.0, 2.0}) {
          std::vector<double> t2 = th;
          t2[j] = th[j] + s * h;
          w.push_back(yderiv4(make<D>(t2.data()), y, J, K - 1));
        }
        const int pl = position_of(4, lower);
        INFO("dtheta: J = " << J << ", K = " << K << ", component " << position_of(4, idx) << ", y = " << y);
        CHECK(std::fabs(exact[position_of(4, idx)] - fd[pl]) <= rel * scale + 1e-12 + stencil_floor(w, pl, h));
      }
    }
}

// ---------------------------------------------------------------- the references

struct ref_row {
  int kind, law, K, pos;
  double y, hi, lo, mu, lam, p, V;
};

std::vector<std::string> split_csv(const char* line) {
  std::vector<std::string> f;
  std::string cur;
  for (const char* p = line; *p && *p != '\n' && *p != '\r'; ++p) {
    if (*p == ',') {
      f.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(*p);
    }
  }
  f.push_back(cur);
  return f;
}

std::vector<ref_row> read_reference(const char* name) {
  std::vector<ref_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if (s.size() != 11) continue;
    auto num = [&](int i) { return std::strtod(s[i].c_str(), nullptr); };
    auto in = [&](int i) { return std::atoi(s[i].c_str()); };
    ref_row r;
    r.kind = in(0), r.law = in(1), r.K = in(2), r.pos = in(3);
    r.y = num(4), r.hi = num(5), r.lo = num(6), r.mu = num(7), r.lam = num(8), r.p = num(9), r.V = num(10);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

double ulp_of(double v) { return v == 0.0 ? std::ldexp(1.0, -1074) : std::ldexp(1.0, std::ilogb(v) - 52); }

// The parameter sets used by the difference checks: (mu, lambda, p, V), p the concavity
// gamma of MCP and the shape a of SCAD, so it is above one for both here.
const std::vector<std::array<double, 4>> CASES = {{0.5, 1.3, 3.0, 2.0},
                                                  {0.0, 0.2, 7.0, 1.0},
                                                  {-2.0, 4.0, 1.25, 9.0},
                                                  {0.0, 3.0, 1.1, 0.5}};

// The bounds of the reference checks, in ulp of the largest component of the order at
// that point. Measured on 2026-11-18 on the gcc build (Serial and OpenMP) and on the
// CUDA build (RTX 4070 SUPER), which agree to the digit here (the reference case runs on
// the host), then multiplied by about one and a half: 3.66, 11.4, 11.5, 8.75, 11.2, 41.7
// and 162 ulp for the orders 0 to 6, 2.46 ulp of max(|log F|, 1) for the cumulative
// functions and 1.33 ulp for Z. The orders rise because log Z goes through the
// multi-index recursions, whose conditioning at a small lambda is what the sixth
// derivative in p sees.
const double BOUND_LOGPDF[7] = {6.0, 18.0, 18.0, 14.0, 18.0, 64.0, 250.0};
const double BOUND_CDF = 4.0;
const double BOUND_Z = 2.0;

double detail_z_mcp(const mcp& d) { return ld::detail_mcp::z_value(d.lambda, d.gamma, d.tail_sigma2); }

}  // namespace

TEST_CASE("mcp and scad: descriptors") {
  CHECK(mcp::kind == ld::distrib_kind::continuous);
  CHECK(scad::kind == ld::distrib_kind::continuous);
  CHECK(mcp::n_params == 4);
  CHECK(scad::n_params == 4);
  CHECK(mcp::y_regularity == 0);
  CHECK(scad::y_regularity == 0);
  CHECK(std::string(mcp::params[2].name) == "gamma");
  CHECK(std::string(scad::params[2].name) == "a");
  CHECK(std::string(mcp::params[3].name) == "tail_sigma2");
  CHECK(std::string(scad::params[3].name) == "tail_sigma2");
  CHECK(mcp::params[1].lower == 0.0);
  CHECK(mcp::params[2].lower == 0.0);
  CHECK(scad::params[2].lower == 1.0);
  CHECK(mcp::params[3].lower == 0.0);
  for (int j = 0; j < 4; ++j) {
    CHECK(mcp::params[j].moves_support == false);
    CHECK(scad::params[j].moves_support == false);
    CHECK(mcp::params[j].circular == false);
  }
  // mu is C^0 in the density and C^1 in log F; lambda and the shape are C^1 at the
  // knots; the tail variance is smooth everywhere
  CHECK(mcp::params[0].regularity == 0);
  CHECK(mcp::params[1].regularity == 1);
  CHECK(mcp::params[2].regularity == 1);
  CHECK(mcp::params[3].regularity == ld::infinitely_smooth);
  CHECK(scad::params[1].regularity == 1);
  const mcp d{0.5, 1.3, 3.0, 2.0};
  const scad e{0.5, 1.3, 3.7, 2.0};
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.5);
  CHECK(e.center() == 0.5);
  CHECK(d.scale() > 0.0);
  CHECK(e.scale() > 0.0);
}

TEST_CASE("mcp and scad: the non-smooth points in y and in the parameters") {
  const mcp d{0.5, 1.3, 3.0, 2.0};
  const scad e{0.5, 1.3, 3.7, 2.0};
  // in y: the kink at the location and every knot, sorted, at the least order
  CHECK(d.y_kinks().count == 3);
  CHECK(d.y_kinks().point[0] == 0.5 - 3.0 * 1.3);
  CHECK(d.y_kinks().point[1] == 0.5);
  CHECK(d.y_kinks().point[2] == 0.5 + 3.0 * 1.3);
  CHECK(d.y_kinks().continuous_order == 0);
  CHECK(e.y_kinks().count == 5);
  CHECK(e.y_kinks().point[0] == 0.5 - 3.7 * 1.3);
  CHECK(e.y_kinks().point[1] == 0.5 - 1.3);
  CHECK(e.y_kinks().point[2] == 0.5);
  CHECK(e.y_kinks().point[3] == 0.5 + 1.3);
  CHECK(e.y_kinks().point[4] == 0.5 + 3.7 * 1.3);
  CHECK(e.y_kinks().continuous_order == 0);
  // in mu: the same point, placed by y
  const double y = 2.9, u = y - 0.5;
  CHECK(d.param_kinks(0, y).count == 1);
  CHECK(d.param_kinks(0, y).point[0] == y);
  CHECK(d.param_kinks(0, y).continuous_order == 0);
  // in lambda: the knot u = gamma lambda for MCP, both knots for SCAD
  CHECK(d.param_kinks(1, y).count == 1);
  CHECK(d.param_kinks(1, y).point[0] == doctest::Approx(u / 3.0));
  CHECK(d.param_kinks(1, y).continuous_order == 1);
  CHECK(e.param_kinks(1, y).count == 2);
  CHECK(e.param_kinks(1, y).point[0] == doctest::Approx(u / 3.7));
  CHECK(e.param_kinks(1, y).point[1] == doctest::Approx(u));
  // in the shape: the knot that moves with it
  CHECK(d.param_kinks(2, y).count == 1);
  CHECK(d.param_kinks(2, y).point[0] == doctest::Approx(u / 1.3));
  CHECK(e.param_kinks(2, y).count == 1);
  CHECK(e.param_kinks(2, y).point[0] == doctest::Approx(u / 1.3));
  // in the tail variance: none; and none at all at y = mu, where no knot is reachable
  CHECK(d.param_kinks(3, y).count == 0);
  CHECK(e.param_kinks(3, y).count == 0);
  CHECK(d.param_kinks(1, 0.5).count == 0);
  CHECK(d.param_kinks(2, 0.5).count == 0);
  // a < 1 is outside SCAD's domain, so the second knot has no kink below u = lambda
  CHECK(e.param_kinks(2, 0.5 + 0.5 * 1.3).count == 0);
}

TEST_CASE("mcp and scad: the inner region is the original penalty, exactly") {
  // The properization replaces the flat tail and leaves the penalty untouched where the
  // original is not flat, so q and its derivatives there are those of mcp_penalty and
  // scad_penalty. logpdf_dy_dk<J, 0> carries no log Z, so the comparison is exact.
  lp::mcp_penalty pm{1};
  lp::scad_penalty ps{1};
  constexpr double eps = std::numeric_limits<double>::epsilon();
  double worst = 0.0;
  for (const auto& c : CASES) {
    const double lam = c[1], p = c[2], V = c[3];
    const double theta[2] = {lam, p};
    const auto sm = pm.theta_state(theta, nullptr);
    const auto ss = ps.theta_state(theta, nullptr);
    const mcp dm{0.0, lam, p, V};
    const scad ds{0.0, lam, p, V};
    for (double f : {0.05, 0.2, 0.5, 0.8, 0.97, 1.0}) {
      // u inside the inner region, which ends at gamma lambda for MCP and at a lambda
      // for SCAD, the same point here since p plays both roles
      for (int law = 0; law < 2; ++law) {
        const double u = f * p * lam;
        for (double sign : {1.0, -1.0}) {
          const double y = sign * u;
          double out[1];
          // the penalty itself: q is the original one, so the value matches exactly
          const double pen = law == 0 ? pm.value(sm, &y, nullptr) : ps.value(ss, &y, nullptr);
          const double q = law == 0 ? ld::detail_mcp::q_value(u, lam, p, V)
                                    : ld::detail_scad::q_value(u, lam, p, V);
          INFO("law " << law << " y " << y << " q " << q << " pen " << pen);
          CHECK(q == pen);
          worst = std::fmax(worst, std::fabs(q - pen));
          // the first and second derivatives in y
          for (int J = 1; J <= 2; ++J) {
            bool ok;
            if (J == 1) ok = law == 0 ? pm.value_dbeta_dk<1, 0>(sm, &y, nullptr, out, nullptr)
                                      : ps.value_dbeta_dk<1, 0>(ss, &y, nullptr, out, nullptr);
            else ok = law == 0 ? pm.value_dbeta_dk<2, 0>(sm, &y, nullptr, out, nullptr)
                               : ps.value_dbeta_dk<2, 0>(ss, &y, nullptr, out, nullptr);
            REQUIRE(ok);
            const double got = law == 0 ? (J == 1 ? dm.logpdf_dy_dk<1, 0>(y)[0] : dm.logpdf_dy_dk<2, 0>(y)[0])
                                        : (J == 1 ? ds.logpdf_dy_dk<1, 0>(y)[0] : ds.logpdf_dy_dk<2, 0>(y)[0]);
            INFO("law " << law << " J " << J << " y " << y);
            // the value is bit-exact; the derivatives are the same quantity written
            // differently on the two sides (the penalty's q1 of SCAD is
            // (a lambda - u)/(a - 1), this law's is lambda - (u - lambda)/(a - 1)),
            // so they agree to rounding and not to the bit
            CHECK(std::fabs(got + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
            worst = std::fmax(worst, std::fabs(got + out[0]) / std::fmax(std::fabs(out[0]), 1.0) / eps);
          }
          // the mixed derivative in y and in each hyperparameter
          for (int w = 0; w < 2; ++w) {
            const int idx[1] = {w};
            bool ok = law == 0 ? pm.value_dbeta_dk<1, 1>(sm, &y, idx, out, nullptr)
                               : ps.value_dbeta_dk<1, 1>(ss, &y, idx, out, nullptr);
            REQUIRE(ok);
            const auto t = law == 0 ? dm.logpdf_dy_dk<1, 1>(y) : ds.logpdf_dy_dk<1, 1>(y);
            INFO("law " << law << " hyperparameter " << w << " y " << y);
            CHECK(std::fabs(t[w + 1] + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
            worst = std::fmax(worst, std::fabs(t[w + 1] + out[0]) / std::fmax(std::fabs(out[0]), 1.0) / eps);
          }
        }
      }
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad against the original penalties on the inner region: worst difference of the derivatives %.3g eps of max(|value|, 1)\n",
                worst);
}

TEST_CASE("mcp and scad: the conventions at the kink and at the knots") {
  for (const auto& c : CASES) {
    const mcp d{c[0], c[1], c[2], c[3]};
    const scad e{c[0], c[1], c[2], c[3]};
    // at y = mu the odd orders in mu are zero (sign(0) = 0, the mean of the two sides)
    const auto t1 = d.logpdf_d1(c[0]);
    CHECK(t1[0] == 0.0);
    const auto s1 = e.logpdf_d1(c[0]);
    CHECK(s1[0] == 0.0);
    const auto t3 = d.logpdf_d3(c[0]);
    CHECK(t3[0] == 0.0);  // three derivatives in mu
    // the even orders come from the inner piece and agree on the two sides
    const auto lo = d.logpdf_d2(c[0] - 1e-9), hi = d.logpdf_d2(c[0] + 1e-9), at = d.logpdf_d2(c[0]);
    CHECK(at[0] == doctest::Approx(lo[0]).epsilon(1e-6));
    CHECK(at[0] == doctest::Approx(hi[0]).epsilon(1e-6));
    // the value is continuous and symmetric about mu
    for (double u : {1e-12, 0.3, 1.0, 5.0}) {
      CHECK(d.logpdf(c[0] + u) == doctest::Approx(d.logpdf(c[0] - u)).epsilon(1e-15));
      CHECK(e.logpdf(c[0] + u) == doctest::Approx(e.logpdf(c[0] - u)).epsilon(1e-15));
    }
    // at a knot the value and every derivative come from the inner piece: the one-sided
    // limit from inside is reached exactly, the one from outside only to first order
    const double kn[3] = {c[2] * c[1], c[1], c[2] * c[1]};
    for (int which = 0; which < 3; ++which) {
      const bool is_mcp = which == 0;
      const double k = kn[which];
      const double y = c[0] + k;
      const double inside = is_mcp ? d.logpdf(c[0] + k * (1 - 1e-13)) : e.logpdf(c[0] + k * (1 - 1e-13));
      const double atk = is_mcp ? d.logpdf(y) : e.logpdf(y);
      INFO("knot " << which << " at u = " << k);
      CHECK(atk == doctest::Approx(inside).epsilon(1e-12));
      // the first derivative in y is continuous across the knot (C^1), the second jumps
      const double d1in = is_mcp ? d.logpdf_dy_dk<1, 0>(c[0] + k * (1 - 1e-9))[0]
                                 : e.logpdf_dy_dk<1, 0>(c[0] + k * (1 - 1e-9))[0];
      const double d1out = is_mcp ? d.logpdf_dy_dk<1, 0>(c[0] + k * (1 + 1e-9))[0]
                                  : e.logpdf_dy_dk<1, 0>(c[0] + k * (1 + 1e-9))[0];
      CHECK(std::fabs(d1in - d1out) <= 1e-6 * (1.0 + std::fabs(d1in)));
    }
    // q is quadratic in u, so three or more derivatives in y vanish inside a piece
    for (double u : {0.37, 1.3, 7.0}) {
      CHECK(d.logpdf_dy_dk<3, 0>(c[0] + u)[0] == 0.0);
      CHECK(d.logpdf_dy_dk<6, 0>(c[0] + u)[0] == 0.0);
      CHECK(e.logpdf_dy_dk<3, 0>(c[0] + u)[0] == 0.0);
      CHECK(e.logpdf_dy_dk<4, 0>(c[0] + u)[0] == 0.0);
    }
  }
}

TEST_CASE("mcp and scad: derivatives against differences") {
  for (const auto& c : CASES) {
    const std::vector<double> th = {c[0], c[1], c[2], c[3]};
    const double kn = c[2] * c[1];
    // points well inside one piece, on both sides of mu: a stencil in lambda or in the
    // shape moves the knots, so a point near one would straddle two pieces
    for (double f : {0.17, 0.63, 1.7, 5.0}) {
      for (double sign : {1.0, -1.0}) {
        const double y = c[0] + sign * f * kn;
        check_derivs<mcp>(th, y, 2e-6);
        check_derivs<scad>(th, y, 2e-6);
      }
    }
  }
}

TEST_CASE("mcp and scad: mixed (y, theta) derivatives against differences") {
  for (const auto& c : {CASES[0], CASES[3]}) {
    const std::vector<double> th = {c[0], c[1], c[2], c[3]};
    const double kn = c[2] * c[1];
    for (double f : {0.31, 1.4}) {
      const double y = c[0] + f * kn;
      check_y_derivs4<mcp>(th, y, 1e-4 * kn, 2e-6);
      check_y_derivs4<scad>(th, y, 1e-4 * kn, 2e-6);
    }
  }
}

TEST_CASE("mcp and scad: the density integrates to one, with and without the knots as cuts") {
  // The quadrature takes the cuts it is given. y_kinks() declares the kink at the
  // location only, the knots being C^1 points; this case measures what that costs.
  double worst_kink = 0.0, worst_all = 0.0;
  for (const auto& c : CASES) {
    const mcp d{c[0], c[1], c[2], c[3]};
    const scad e{c[0], c[1], c[2], c[3]};
    const double km[3] = {c[0], c[0] - c[2] * c[1], c[0] + c[2] * c[1]};
    const double ks[5] = {c[0],          c[0] - c[1],          c[0] + c[1],
                          c[0] - c[2] * c[1], c[0] + c[2] * c[1]};
    for (int law = 0; law < 2; ++law) {
      auto f = [&](double y) { return law == 0 ? std::exp(d.logpdf(y)) : std::exp(e.logpdf(y)); };
      const double ctr = law == 0 ? d.center() : e.center();
      const double sc = law == 0 ? d.scale() : e.scale();
      const double m1 = laplace::math::quad::panels_quad(f, -ld::distrib_infinity, ld::distrib_infinity, ctr,
                                                         sc, km, 1).value;
      const double m2 = laplace::math::quad::panels_quad(f, -ld::distrib_infinity, ld::distrib_infinity, ctr,
                                                         sc, law == 0 ? km : ks, law == 0 ? 3 : 5).value;
      INFO("law " << law << " lambda " << c[1] << " p " << c[2]);
      worst_kink = std::fmax(worst_kink, std::fabs(m1 - 1.0));
      worst_all = std::fmax(worst_all, std::fabs(m2 - 1.0));
      // What the knots are worth as cuts, measured on 2026-11-18: with the kink at the
      // location alone the panels straddle the knots, where the second derivative of
      // the density jumps, and the mass is out by 8.7e-11; with every knot as a cut it
      // is out by 4.4e-16.
      CHECK(std::fabs(m1 - 1.0) <= 2e-10);
      CHECK(std::fabs(m2 - 1.0) <= 1e-15);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad mass: worst |mass - 1| with the kink alone %.3g, with every knot %.3g\n", worst_kink,
                worst_all);
}

TEST_CASE("mcp and scad: the closed cumulative functions") {
  for (const auto& c : CASES) {
    const mcp d{c[0], c[1], c[2], c[3]};
    const scad e{c[0], c[1], c[2], c[3]};
    // the median is the location
    CHECK(d.cdf(c[0]) == doctest::Approx(0.5).epsilon(1e-14));
    CHECK(e.cdf(c[0]) == doctest::Approx(0.5).epsilon(1e-14));
    CHECK(d.ccdf(c[0]) == doctest::Approx(0.5).epsilon(1e-14));
    const double kn = c[2] * c[1];
    for (double f : {-6.0, -1.5, -0.4, 0.4, 1.5, 6.0}) {
      const double y = c[0] + f * kn;
      for (int law = 0; law < 2; ++law) {
        const double F = law == 0 ? d.cdf(y) : e.cdf(y);
        const double S = law == 0 ? d.ccdf(y) : e.ccdf(y);
        const double lF = law == 0 ? d.log_cdf(y) : e.log_cdf(y);
        const double lS = law == 0 ? d.log_ccdf(y) : e.log_ccdf(y);
        INFO("law " << law << " y " << y);
        CHECK(F + S == doctest::Approx(1.0).epsilon(1e-14));
        CHECK(std::log(F) == doctest::Approx(lF).epsilon(1e-13));
        CHECK(std::log(S) == doctest::Approx(lS).epsilon(1e-13));
        // F against the quadrature of the density, cut at the kink and at every knot
        const double cm[3] = {c[0], c[0] - c[2] * c[1], c[0] + c[2] * c[1]};
        const double cs[5] = {c[0],               c[0] - c[1],          c[0] + c[1],
                              c[0] - c[2] * c[1], c[0] + c[2] * c[1]};
        const double q = laplace::math::quad::panels_quad(
                             [&](double t) { return law == 0 ? std::exp(d.logpdf(t)) : std::exp(e.logpdf(t)); },
                             -ld::distrib_infinity, y, c[0], law == 0 ? d.scale() : e.scale(),
                             law == 0 ? cm : cs, law == 0 ? 3 : 5)
                             .value;
        CHECK(F == doctest::Approx(q).epsilon(1e-11));
      }
    }
  }
}

TEST_CASE("mcp and scad: the normal limit") {
  // MCP as gamma -> 0 and SCAD as lambda -> 0 lose their inner region and leave the
  // normal of variance V: q(u) -> u^2/(2 V) and Z -> sqrt(2 pi V).
  const double V = 2.0;
  const mcp d{0.4, 1.3, 1e-14, V};
  const scad e{0.4, 1e-14, 3.7, V};
  const ld::gaussian2_distrib g{0.4, V};
  double worst = 0.0;
  for (double y = -5.0; y <= 5.0; y += 0.25) {
    worst = std::fmax(worst, std::fabs(d.logpdf(y) - g.logpdf(y)));
    worst = std::fmax(worst, std::fabs(e.logpdf(y) - g.logpdf(y)));
    INFO("y = " << y);
    CHECK(d.logpdf(y) == doctest::Approx(g.logpdf(y)).epsilon(1e-10));
    CHECK(e.logpdf(y) == doctest::Approx(g.logpdf(y)).epsilon(1e-10));
    CHECK(d.cdf(y) == doctest::Approx(g.cdf(y)).epsilon(1e-10));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad normal limit: worst absolute difference of the log-density %.3g\n", worst);
}

TEST_CASE("mcp and scad: the Laplace limit") {
  // When the knot is far out the tail carries no mass and the normalizing constant tends
  // to the Laplace's 2/lambda: with x = lambda sqrt(gamma/2), Dawson's asymptotic
  // expansion gives sqrt(2 gamma) F(x) = (1/lambda)(1 + 1/(2 x^2) + ...), so
  //   log f - log f_Laplace = u^2/(2 gamma) - log(lambda Z/2),
  // the first term the inner curvature of the penalty, which stays, the second of size
  // 1/(2 x^2). The identity is what is checked here, to the next term of the expansion.
  for (double lam : {30.0, 300.0}) {
    const double gam = 3.0;
    const mcp d{0.4, lam, gam, 1.0};
    const ld::laplace2_distrib l{0.4, lam};
    const double x = lam * std::sqrt(0.5 * gam);
    const double zgap = std::log(lam * detail_z_mcp(d) / 2.0);
    double worst = 0.0;
    for (double u = 0.0; u <= 0.1; u += 0.01) {
      const double got = d.logpdf(0.4 + u) - l.logpdf(0.4 + u);
      const double want = u * u / (2.0 * gam) - zgap;
      worst = std::fmax(worst, std::fabs(got - want));
      INFO("lambda = " << lam << ", u = " << u);
      CHECK(std::fabs(got - want) <= 1e-13);
    }
    // the gap in the constant is 1/(2 x^2) to the next order
    CHECK(std::fabs(zgap - 0.5 / (x * x)) <= 4.0 / (x * x * x * x));
    if (std::getenv("LAPLACE_PRINT_ERRORS"))
      std::printf("mcp Laplace limit at lambda %g: worst %.3g on the identity; log(lambda Z/2) %.4g against 1/(2 x^2) %.4g\n",
                  lam, worst, zgap, 0.5 / (x * x));
  }
}

TEST_CASE("mcp and scad: the generic quantile and the generic cdf route") {
  double worst = 0.0;
  for (const auto& c : {CASES[0], CASES[2]}) {
    const mcp d{c[0], c[1], c[2], c[3]};
    const scad e{c[0], c[1], c[2], c[3]};
    for (double p : {1e-8, 1e-3, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0 - 1e-3, 1.0 - 1e-8}) {
      for (int law = 0; law < 2; ++law) {
        const double y = law == 0 ? ld::distrib_quantile(d, p) : ld::distrib_quantile(e, p);
        const double back = law == 0 ? d.cdf(y) : e.cdf(y);
        INFO("law " << law << " p " << p << " quantile " << y);
        const double rel = std::fabs(back - p) / p;
        worst = std::fmax(worst, rel);
        CHECK(rel <= 1e-10);
      }
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad generic quantile: worst relative error of cdf(quantile(p)) against p %.3g\n", worst);
}

TEST_CASE("mcp and scad: start values") {
  std::vector<double> y;
  for (int i = 0; i < 101; ++i) y.push_back(-2.0 + 0.04 * i);
  const auto s = ld::summarize_sample(y);
  const auto a = ld::distrib_start_value(mcp{0.0, 1.0, 3.0, 1.0}, s);
  CHECK(a.lambda > 0.0);
  CHECK(a.gamma > 0.0);
  CHECK(a.tail_sigma2 > 0.0);
  const auto b = ld::distrib_start_value(scad{0.0, 1.0, 3.7, 1.0}, s);
  CHECK(b.a > 1.0);
  CHECK(b.tail_sigma2 > 0.0);
}

TEST_CASE("mcp and scad: the generic expected information") {
  // The score form, which is the one that holds where a parameter's regularity is
  // finite: the hessian form misses the point mass at the kink, and the route marks its
  // (mu, mu) component invalid for that reason. Both forms need every knot among the
  // cuts: with the kink at the location alone the panel quadrature does not converge on
  // any component (measured 2026-11-18, all ten NaN in both forms).
  for (const auto& c : {CASES[0], CASES[1]}) {
    const mcp d{c[0], c[1], c[2], c[3]};
    const scad e{c[0], c[1], c[2], c[3]};
    const auto Im = ld::distrib_expected_info<0>(d);
    const auto Is = ld::distrib_expected_info<0>(e);
    for (int i = 0; i < ld::sym_tensor<4, 2>::size; ++i) {
      INFO("component " << i << " lambda " << c[1] << " p " << c[2]);
      CHECK(std::isfinite(Im[i]));
      CHECK(std::isfinite(Is[i]));
    }
    // the diagonal is positive and the location is orthogonal to the other three, the
    // density being symmetric about mu
    CHECK(Im[0] > 0.0);
    CHECK(Im[4] > 0.0);
    CHECK(Im[7] > 0.0);
    CHECK(Im[9] > 0.0);
    for (int j : {1, 2, 3}) CHECK(std::fabs(Im[j]) <= 1e-12 * std::sqrt(Im[0] * Im[4]));
    for (int j : {1, 2, 3}) CHECK(std::fabs(Is[j]) <= 1e-12 * std::sqrt(Is[0] * Is[4]));
  }
}

TEST_CASE("mcp and scad: the log-density, the cumulative functions and Z against mpmath") {
  // tools/reference/distrib_mcp_scad.py: SymPy's derivatives of the definition evaluated
  // by mpmath at 300 and 400 bits, with the closed normalizing constant pinned against
  // the quadrature of the definition in the same script.
  const auto rows = read_reference("distrib_mcp_scad.csv");
  CHECK(rows.size() > 2000);
  std::map<std::tuple<int, int, double, double, double, double, int>, double> scale;
  for (const auto& r : rows) {
    if (r.kind != 0) continue;
    auto& s = scale[std::make_tuple(r.law, r.K, r.y, r.lam, r.p, r.V, 0)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[7] = {}, worst_cdf = 0.0, worst_cdf_rel = 0.0, worst_z = 0.0;
  std::tuple<double, double, double, double, int> where[7];
  for (const auto& r : rows) {
    const mcp d{r.mu, r.lam, r.p, r.V};
    const scad e{r.mu, r.lam, r.p, r.V};
    if (r.kind == 3) {
      const double z = std::exp(r.law == 0 ? ld::detail_mcp::log_z(r.lam, r.p, r.V)
                                           : ld::detail_scad::log_z(r.lam, r.p, r.V));
      const double err = std::fabs((z - r.hi) - r.lo) / ulp_of(r.hi);
      worst_z = std::fmax(worst_z, err);
      INFO("Z: law " << r.law << " lambda " << r.lam << " p " << r.p << " V " << r.V << " ulp " << err);
      CHECK(err <= BOUND_Z);
      continue;
    }
    if (r.kind == 1 || r.kind == 2) {
      const double v = r.kind == 1 ? (r.law == 0 ? d.log_cdf(r.y) : e.log_cdf(r.y))
                                   : (r.law == 0 ? d.log_ccdf(r.y) : e.log_ccdf(r.y));
      const double raw = std::fabs((v - r.hi) - r.lo);
      // in ulp of max(|log F|, 1), the measure a reader of a log-likelihood cares
      // about; the relative figure is kept too, since a Gaussian tail probability
      // cannot be computed to better than z^2 ulp, z the standardized distance beyond
      // the knot, the argument of Phi carrying one ulp and log Phi having slope -z
      const double err = raw / ulp_of(std::fmax(std::fabs(r.hi), 1.0));
      worst_cdf = std::fmax(worst_cdf, err);
      worst_cdf_rel = std::fmax(worst_cdf_rel, raw / ulp_of(r.hi));
      INFO("kind " << r.kind << " law " << r.law << " y " << r.y << " ulp1 " << err);
      CHECK(err <= BOUND_CDF);
      continue;
    }
    const auto v = r.law == 0 ? lderiv(d, r.y, r.K) : lderiv(e, r.y, r.K);
    const double got = v[r.pos];
    const double sc = scale[std::make_tuple(r.law, r.K, r.y, r.lam, r.p, r.V, 0)];
    const double err = std::fabs((got - r.hi) - r.lo) / ulp_of(sc);
    if (err > worst[r.K]) {
      worst[r.K] = err;
      where[r.K] = std::make_tuple(r.y, r.lam, r.p, r.V, r.law);
    }
    INFO("law " << r.law << " K " << r.K << " pos " << r.pos << " y " << r.y << " lambda " << r.lam << " p "
                << r.p << " V " << r.V << " ulp " << err);
    CHECK(err <= BOUND_LOGPDF[r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (int K = 0; K <= 6; ++K)
      std::printf("mcp/scad logpdf order %d: max %.3g ulp of the order's scale (y %.6g lambda %.6g p %.6g V %.6g law %d)\n",
                  K, worst[K], std::get<0>(where[K]), std::get<1>(where[K]), std::get<2>(where[K]),
                  std::get<3>(where[K]), std::get<4>(where[K]));
    std::printf("mcp/scad log F and log(1 - F): max %.3g ulp of max(|log F|, 1), %.3g ulp relative; Z: max %.3g ulp\n",
                worst_cdf, worst_cdf_rel, worst_z);
  }
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE (doctest gives each case an internal-linkage function whose number restarts
// per translation unit, and nvcc tags every extended lambda with that address). One
// kernel per order, the stack frame of a kernel being that of its largest call: the
// tables of log Z are the largest frame here, and at order six they are what the
// triangular packing of detail/properized_nonconvex.hpp was introduced for.
template <int K>
void mcp_scad_kernel_order(double lam, double p, double V, double bound) {
  const int n = 40;
  Kokkos::View<double*> out("out", n), mag("mag", n);
  Kokkos::parallel_for(
      "mcp_scad_order", n, KOKKOS_LAMBDA(const int i) {
        const double y = -6.0 + 0.3 * i;
        const ld::mcp_distrib a{0.4, lam, p, V};
        const ld::scad_distrib b{0.4, lam, p, V};
        const auto t = a.logpdf_dk<K>(y);
        const auto s = b.logpdf_dk<K>(y);
        double v = 0.0, m = 0.0;
        for (int j = 0; j < ld::sym_tensor<4, K>::size; ++j) {
          v += t[j] + s[j];
          m += Kokkos::fabs(t[j]) + Kokkos::fabs(s[j]);
        }
        out(i) = v;
        mag(i) = m;
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -6.0 + 0.3 * i;
    const mcp a{0.4, lam, p, V};
    const scad b{0.4, lam, p, V};
    const auto t = a.logpdf_dk<K>(y);
    const auto s = b.logpdf_dk<K>(y);
    double v = 0.0, m = 0.0;
    for (int j = 0; j < ld::sym_tensor<4, K>::size; ++j) {
      v += t[j] + s[j];
      m += std::fabs(t[j]) + std::fabs(s[j]);
    }
    worst = std::fmax(worst, std::fabs(h(i) - v) / m);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad kernel order %d (lambda %g p %g): worst host-device difference %.3g of the sum of |components|\n",
                K, lam, p, worst);
  CHECK(worst <= bound);
}

void mcp_scad_kernel_other(double lam, double p, double V) {
  const int n = 16;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "mcp_scad_other", n, KOKKOS_LAMBDA(const int i) {
        const double y = -6.0 + 0.8 * i;
        const ld::mcp_distrib a{0.4, lam, p, V};
        const ld::scad_distrib b{0.4, lam, p, V};
        out(i) = a.logpdf(y) + b.logpdf(y) + a.log_cdf(y) + b.log_ccdf(y) + a.cdf(y) + b.cdf(y) +
                 a.logpdf_dy_dk<2, 2>(y)[3] + b.logpdf_dy_dk<1, 1>(y)[2];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -6.0 + 0.8 * i;
    const mcp a{0.4, lam, p, V};
    const scad b{0.4, lam, p, V};
    const double r = a.logpdf(y) + b.logpdf(y) + a.log_cdf(y) + b.log_ccdf(y) + a.cdf(y) + b.cdf(y) +
                     a.logpdf_dy_dk<2, 2>(y)[3] + b.logpdf_dy_dk<1, 1>(y)[2];
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad kernel (value, cumulative, mixed): worst relative difference %.3g\n", worst);
  CHECK(worst <= 1e-15);
}

void mcp_scad_kernel_case() {
  // the worst host-device differences measured on 2026-11-18 on the RTX 4070 SUPER are
  // 1.18e-15 at order one and between 2.1e-16 and 5.5e-16 at the others, so one bound of
  // 2e-15 covers them all
  mcp_scad_kernel_order<1>(1.3, 3.0, 2.0, 2e-15);
  mcp_scad_kernel_order<2>(1.3, 3.0, 2.0, 2e-15);
  mcp_scad_kernel_order<3>(1.3, 3.0, 2.0, 2e-15);
  mcp_scad_kernel_order<4>(1.3, 3.0, 2.0, 2e-15);
  mcp_scad_kernel_order<5>(1.3, 3.0, 2.0, 2e-15);
  mcp_scad_kernel_order<6>(1.3, 3.0, 2.0, 2e-15);
  mcp_scad_kernel_order<2>(0.2, 7.0, 1.0, 2e-15);
  mcp_scad_kernel_order<6>(0.2, 7.0, 1.0, 2e-15);
  mcp_scad_kernel_other(1.3, 3.0, 2.0);
}

TEST_CASE("mcp and scad: every order inside a kernel") { mcp_scad_kernel_case(); }
