// The test of mcp_distrib and scad_distrib (include/laplace/distrib/mcp.hpp, scad.hpp)
// with the Student and the logistic properizing tails
// (include/laplace/distrib/properizing_tail.hpp): descriptors, the inner region against
// the original mcp_penalty and scad_penalty, which the choice of tail cannot touch, the
// conventions at the kink and at the knots, every derivative order against differences,
// the mixed (y, theta) derivatives, the mass by quadrature, the closed cumulative
// functions, the elementary constants of the logistic tail and the Cauchy case of the
// Student one, the Gaussian tail as the Student tail's limit in its degrees of freedom,
// the references of tools/reference/distrib_mcp_scad.py, and a kernel launch per order.
// The Gaussian tail is in tests/distrib/test_mcp_scad.cpp.
#include <doctest/doctest.h>

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/mcp.hpp>
#include <laplace/distrib/scad.hpp>
#include <laplace/math/quadrature.hpp>
#include <laplace/penalty/nonconvex_penalty.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"

namespace laplace_distrib_check {
// brace elision carries the trailing parameters into the tail
template <>
ld::mcp_distrib<ld::student_tail> make<ld::mcp_distrib<ld::student_tail>>(const double* t) {
  return {t[0], t[1], t[2], t[3], t[4]};
}
template <>
ld::scad_distrib<ld::student_tail> make<ld::scad_distrib<ld::student_tail>>(const double* t) {
  return {t[0], t[1], t[2], t[3], t[4]};
}
template <>
ld::mcp_distrib<ld::logistic_tail> make<ld::mcp_distrib<ld::logistic_tail>>(const double* t) {
  return {t[0], t[1], t[2], t[3]};
}
template <>
ld::scad_distrib<ld::logistic_tail> make<ld::scad_distrib<ld::logistic_tail>>(const double* t) {
  return {t[0], t[1], t[2], t[3]};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

using mcp_s = ld::mcp_distrib<ld::student_tail>;
using mcp_l = ld::mcp_distrib<ld::logistic_tail>;
using mcp_g = ld::mcp_distrib<ld::gaussian_tail>;
using scad_s = ld::scad_distrib<ld::student_tail>;
using scad_l = ld::scad_distrib<ld::logistic_tail>;
using scad_g = ld::scad_distrib<ld::gaussian_tail>;
namespace lp = laplace::penalty;

// --------------------------------------------------------- the y-derivatives over P parameters

// check_common's yderiv packs two parameters; these laws carry four or five, so the
// switch is written once here, templated on their number
#define YP(J, K) \
  case J * 10 + K: return as_vec<P, K>(d.template logpdf_dy_dk<J, K>(y))
template <int P, class D>
std::vector<double> yderiv_p(const D& d, double y, int J, int K) {
  switch (J * 10 + K) {
    YP(0, 0); YP(0, 1); YP(0, 2); YP(0, 3); YP(0, 4); YP(0, 5); YP(0, 6);
    YP(1, 0); YP(1, 1); YP(1, 2); YP(1, 3); YP(1, 4); YP(1, 5);
    YP(2, 0); YP(2, 1); YP(2, 2); YP(2, 3); YP(2, 4);
    YP(3, 0); YP(3, 1); YP(3, 2); YP(3, 3);
    YP(4, 0); YP(4, 1); YP(4, 2);
    YP(5, 0); YP(5, 1);
    YP(6, 0);
    default: return {};
  }
}
#undef YP

// every (J, K): against the difference in y of (J - 1, K) and in each parameter of (J, K - 1)
template <class D>
void check_y_derivs_p(const std::vector<double>& th, double y, double hy, double rel) {
  constexpr int P = D::n_params;
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yderiv_p<P>(make<D>(th.data()), y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0})
        v.push_back(yderiv_p<P>(make<D>(th.data()), y + s * hy, J - 1, K));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        INFO("dy: J = " << J << ", K = " << K << ", component " << i << ", y = " << y);
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12 + stencil_floor(v, i, hy));
      }
      if (K == 0) continue;
      for (const auto& idx : sorted_indices(P, K)) {
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const int j = idx.back();
        const double h = 1e-3 * std::fmax(1e-3, std::fabs(th[j]));
        const auto fd =
            diff5([&](const double* t) { return yderiv_p<P>(make<D>(t), y, J, K - 1); }, th, j, h);
        std::vector<std::vector<double>> w;
        for (double s : {-2.0, -1.0, 1.0, 2.0}) {
          std::vector<double> t2 = th;
          t2[j] = th[j] + s * h;
          w.push_back(yderiv_p<P>(make<D>(t2.data()), y, J, K - 1));
        }
        const int pl = position_of(P, lower);
        INFO("dtheta: J = " << J << ", K = " << K << ", component " << position_of(P, idx)
                            << ", y = " << y);
        CHECK(std::fabs(exact[position_of(P, idx)] - fd[pl]) <=
              rel * scale + 1e-12 + stencil_floor(w, pl, h));
      }
    }
}

// ---------------------------------------------------------------- the references

struct ref_row {
  int kind, law, tail, K, pos;
  double y, hi, lo, mu, lam, p, t1, t2;
};
constexpr int REF_FIELDS = 13;

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

// the rows of the Student (tail 1) and the logistic (tail 2) tails
std::vector<ref_row> read_reference(const char* name) {
  std::vector<ref_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if ((int)s.size() != REF_FIELDS) continue;
    auto num = [&](int i) { return std::strtod(s[i].c_str(), nullptr); };
    auto in = [&](int i) { return std::atoi(s[i].c_str()); };
    if (in(2) == 0) continue;  // the Gaussian tail has its own test
    ref_row r;
    r.kind = in(0), r.law = in(1), r.tail = in(2), r.K = in(3), r.pos = in(4);
    r.y = num(5), r.hi = num(6), r.lo = num(7), r.mu = num(8), r.lam = num(9), r.p = num(10),
    r.t1 = num(11), r.t2 = num(12);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

double ulp_of(double v) { return v == 0.0 ? std::ldexp(1.0, -1074) : std::ldexp(1.0, std::ilogb(v) - 52); }

// The parameter sets of the difference checks: (mu, lambda, p, s, nu) for the Student
// tail and (mu, lambda, p, s) for the logistic one, p the concavity gamma of MCP and the
// shape a of SCAD, so it is above one for both here.
const std::vector<std::array<double, 5>> CASES_S = {{0.5, 1.3, 3.0, 1.1, 5.0},
                                                    {0.0, 0.2, 7.0, 1.0, 1.0},
                                                    {-2.0, 4.0, 1.25, 3.0, 0.7},
                                                    {0.0, 3.0, 1.1, 0.5, 40.0}};
const std::vector<std::array<double, 4>> CASES_L = {{0.5, 1.3, 3.0, 1.1},
                                                    {0.0, 0.2, 7.0, 1.0},
                                                    {-2.0, 4.0, 1.25, 3.0},
                                                    {0.0, 3.0, 1.1, 0.5}};

// The bounds of the reference checks, in ulp of the largest component of the order at
// that point. Measured on 2026-11-19 over 74184 values, then multiplied by about one and
// a half: 2, 6, 6, 10, 9, 25 and 163 ulp for the orders 0 to 6, 2 ulp of
// max(|log F|, 1) for the cumulative functions and 2 ulp for Z. The reference case runs
// on the host in both builds, so the gcc and the CUDA figures are the same. The orders
// rise because log Z goes through the multi-index recursions and the tail's components
// through the Stirling expansion of its parameters, whose conditioning at a small lambda
// is what the sixth derivative sees.
const double BOUND_LOGPDF[7] = {4.0, 10.0, 10.0, 16.0, 16.0, 40.0, 250.0};
const double BOUND_CDF = 4.0;
const double BOUND_Z = 3.0;
// The bound of the prox against mpmath, in ulp of the minimizer. Measured on 2026-11-20
// over 12075 rows with a well separated minimum: 13 ulp, at the scad knot with a step of
// 20, where the minimizer is the knot's neighbourhood and the candidates' objectives
// differ in their last digits. The 168 rows whose objective is flat over a whole piece
// are checked on the objective instead, as is every row, against the reference's point.
const double BOUND_PROX = 20.0;

}  // namespace

TEST_CASE("mcp and scad tails: descriptors") {
  CHECK(mcp_s::n_params == 5);
  CHECK(scad_s::n_params == 5);
  CHECK(mcp_l::n_params == 4);
  CHECK(scad_l::n_params == 4);
  CHECK(ld::student_tail::n_tail_params == 2);
  CHECK(ld::logistic_tail::n_tail_params == 1);
  CHECK(ld::gaussian_tail::n_tail_params == 1);
  // the law's own three parameters come first, then the tail's
  CHECK(std::string(mcp_s::params[0].name) == "mu");
  CHECK(std::string(mcp_s::params[1].name) == "lambda");
  CHECK(std::string(mcp_s::params[2].name) == "gamma");
  CHECK(std::string(mcp_s::params[3].name) == "tail_scale");
  CHECK(std::string(mcp_s::params[4].name) == "tail_df");
  CHECK(std::string(scad_s::params[2].name) == "a");
  CHECK(std::string(mcp_l::params[3].name) == "tail_scale");
  for (int j = 3; j < 5; ++j) {
    CHECK(mcp_s::params[j].lower == 0.0);
    CHECK(mcp_s::params[j].regularity == ld::infinitely_smooth);
  }
  CHECK(mcp_s::kind == ld::distrib_kind::continuous);
  CHECK(mcp_s::y_regularity == 0);
  CHECK(scad_s::y_regularity == 0);
  // the knots in y do not depend on the tail
  const mcp_s a{0.5, 1.3, 3.0, 1.1, 5.0};
  const mcp_g b{0.5, 1.3, 3.0, 1.21};
  CHECK(a.y_kinks().count == 3);
  CHECK(a.y_kinks().count == b.y_kinks().count);
  for (int i = 0; i < 3; ++i) CHECK(a.y_kinks().point[i] == b.y_kinks().point[i]);
  const scad_s c{0.5, 1.3, 3.7, 1.1, 5.0};
  CHECK(c.y_kinks().count == 5);
  // the tail is smooth in its own parameters, so they declare no kink
  CHECK(a.param_kinks(3, 2.0).count == 0);
  CHECK(a.param_kinks(4, 2.0).count == 0);
  CHECK(c.param_kinks(3, 2.0).count == 0);
}

TEST_CASE("mcp and scad tails: the inner region is the original penalty, exactly") {
  // The choice of tail changes nothing on the inner pieces, where q is still the
  // original penalty: the value is bit-exact against mcp_penalty and scad_penalty for
  // every tail, as it is for the Gaussian one.
  lp::mcp_penalty pm{1};
  lp::scad_penalty ps{1};
  constexpr double eps = std::numeric_limits<double>::epsilon();
  double worst = 0.0;
  for (const auto& c : CASES_S) {
    const double lam = c[1], p = c[2], s = c[3], nu = c[4];
    const double theta[2] = {lam, p};
    const auto sm = pm.theta_state(theta, nullptr);
    const auto ss = ps.theta_state(theta, nullptr);
    const mcp_s dm{0.0, lam, p, s, nu};
    const scad_s ds{0.0, lam, p, s, nu};
    const mcp_l em{0.0, lam, p, s};
    const scad_l es{0.0, lam, p, s};
    for (double f : {0.05, 0.2, 0.5, 0.8, 0.97, 1.0}) {
      const double u = f * p * lam;  // inside the inner region, which ends at p lambda
      for (double sign : {1.0, -1.0}) {
        const double y = sign * u;
        const double pen_m = pm.value(sm, &y, nullptr);
        const double pen_s = ps.value(ss, &y, nullptr);
        CHECK(ld::detail_mcp::q_value(u, lam, p, dm.tail) == pen_m);
        CHECK(ld::detail_mcp::q_value(u, lam, p, em.tail) == pen_m);
        CHECK(ld::detail_scad::q_value(u, lam, p, ds.tail) == pen_s);
        CHECK(ld::detail_scad::q_value(u, lam, p, es.tail) == pen_s);
        // the first two derivatives in y, the same quantity written differently on the
        // two sides, so they agree to rounding and not to the bit
        double out[1];
        for (int J = 1; J <= 2; ++J) {
          bool ok = J == 1 ? pm.value_dbeta_dk<1, 0>(sm, &y, nullptr, out, nullptr)
                           : pm.value_dbeta_dk<2, 0>(sm, &y, nullptr, out, nullptr);
          REQUIRE(ok);
          const double g1 = J == 1 ? dm.logpdf_dy_dk<1, 0>(y)[0] : dm.logpdf_dy_dk<2, 0>(y)[0];
          const double g2 = J == 1 ? em.logpdf_dy_dk<1, 0>(y)[0] : em.logpdf_dy_dk<2, 0>(y)[0];
          INFO("mcp J " << J << " y " << y);
          CHECK(std::fabs(g1 + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
          CHECK(std::fabs(g2 + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
          worst = std::fmax(worst, std::fabs(g1 + out[0]) / std::fmax(std::fabs(out[0]), 1.0) / eps);
          ok = J == 1 ? ps.value_dbeta_dk<1, 0>(ss, &y, nullptr, out, nullptr)
                      : ps.value_dbeta_dk<2, 0>(ss, &y, nullptr, out, nullptr);
          REQUIRE(ok);
          const double h1 = J == 1 ? ds.logpdf_dy_dk<1, 0>(y)[0] : ds.logpdf_dy_dk<2, 0>(y)[0];
          const double h2 = J == 1 ? es.logpdf_dy_dk<1, 0>(y)[0] : es.logpdf_dy_dk<2, 0>(y)[0];
          INFO("scad J " << J << " y " << y);
          CHECK(std::fabs(h1 + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
          CHECK(std::fabs(h2 + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
          worst = std::fmax(worst, std::fabs(h1 + out[0]) / std::fmax(std::fabs(out[0]), 1.0) / eps);
        }
        // the mixed derivative in y and in each hyperparameter of the penalty
        for (int w = 0; w < 2; ++w) {
          const int idx[1] = {w};
          REQUIRE(pm.value_dbeta_dk<1, 1>(sm, &y, idx, out, nullptr));
          const auto t = dm.logpdf_dy_dk<1, 1>(y);
          INFO("mcp hyperparameter " << w << " y " << y);
          CHECK(std::fabs(t[w + 1] + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
          REQUIRE(ps.value_dbeta_dk<1, 1>(ss, &y, idx, out, nullptr));
          const auto q = ds.logpdf_dy_dk<1, 1>(y);
          INFO("scad hyperparameter " << w << " y " << y);
          CHECK(std::fabs(q[w + 1] + out[0]) <= 8.0 * eps * std::fmax(std::fabs(out[0]), 1.0));
        }
        // and the derivatives in the tail's own parameters vanish on the inner pieces
        const auto t1 = dm.logpdf_dy_dk<1, 1>(y);
        CHECK(t1[3] == 0.0);
        CHECK(t1[4] == 0.0);
      }
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad with the Student and logistic tails against the original penalties: worst difference of the derivatives %.3g eps of max(|value|, 1)\n",
                worst);
}

TEST_CASE("mcp and scad tails: the conventions at the kink and at the knots") {
  for (const auto& c : CASES_S) {
    const mcp_s d{c[0], c[1], c[2], c[3], c[4]};
    const scad_s e{c[0], c[1], c[2], c[3], c[4]};
    const mcp_l f{c[0], c[1], c[2], c[3]};
    // at y = mu the odd orders in mu are zero (sign(0) = 0, the mean of the two sides)
    CHECK(d.logpdf_d1(c[0])[0] == 0.0);
    CHECK(e.logpdf_d1(c[0])[0] == 0.0);
    CHECK(f.logpdf_d1(c[0])[0] == 0.0);
    CHECK(d.logpdf_d3(c[0])[0] == 0.0);
    // the value is continuous and symmetric about mu
    for (double u : {1e-12, 0.3, 1.0, 5.0}) {
      CHECK(d.logpdf(c[0] + u) == doctest::Approx(d.logpdf(c[0] - u)).epsilon(1e-15));
      CHECK(e.logpdf(c[0] + u) == doctest::Approx(e.logpdf(c[0] - u)).epsilon(1e-15));
      CHECK(f.logpdf(c[0] + u) == doctest::Approx(f.logpdf(c[0] - u)).epsilon(1e-15));
    }
    // at a knot the value comes from the inner piece and q' is continuous there (C^1)
    const double kn[3] = {c[2] * c[1], c[1], c[2] * c[1]};
    for (int which = 0; which < 3; ++which) {
      const double k = kn[which];
      const double y = c[0] + k;
      const bool is_mcp = which == 0;
      const double inside = is_mcp ? d.logpdf(c[0] + k * (1 - 1e-13)) : e.logpdf(c[0] + k * (1 - 1e-13));
      const double atk = is_mcp ? d.logpdf(y) : e.logpdf(y);
      INFO("knot " << which << " at u = " << k);
      CHECK(atk == doctest::Approx(inside).epsilon(1e-12));
      const double d1in = is_mcp ? d.logpdf_dy_dk<1, 0>(c[0] + k * (1 - 1e-9))[0]
                                 : e.logpdf_dy_dk<1, 0>(c[0] + k * (1 - 1e-9))[0];
      const double d1out = is_mcp ? d.logpdf_dy_dk<1, 0>(c[0] + k * (1 + 1e-9))[0]
                                  : e.logpdf_dy_dk<1, 0>(c[0] + k * (1 + 1e-9))[0];
      CHECK(std::fabs(d1in - d1out) <= 1e-6 * (1.0 + std::fabs(d1in)));
    }
    // inside an inner piece q is quadratic in u, so three derivatives in y vanish; in the
    // tail they do not, which is the whole point of a tail that is not a parabola
    for (double u : {0.3 * c[2] * c[1], 0.9 * c[2] * c[1]}) {
      CHECK(d.logpdf_dy_dk<3, 0>(c[0] + u)[0] == 0.0);
      CHECK(f.logpdf_dy_dk<3, 0>(c[0] + u)[0] == 0.0);
    }
    const double far = c[0] + 2.0 * c[2] * c[1] + 1.0;
    CHECK(d.logpdf_dy_dk<3, 0>(far)[0] != 0.0);
    CHECK(f.logpdf_dy_dk<3, 0>(far)[0] != 0.0);
  }
}

TEST_CASE("mcp and scad tails: derivatives against differences") {
  for (const auto& c : CASES_S) {
    const std::vector<double> ts(c.begin(), c.end());
    const std::vector<double> tl(c.begin(), c.begin() + 4);
    const double knot = c[2] * c[1];
    for (double u : {0.3 * knot, 1.4 * knot, 3.0 * knot + 1.0}) {
      check_derivs<mcp_s>(ts, c[0] + u, 2e-6);
      check_derivs<scad_s>(ts, c[0] + u, 2e-6);
      check_derivs<mcp_l>(tl, c[0] + u, 2e-6);
      check_derivs<scad_l>(tl, c[0] + u, 2e-6);
    }
  }
}

TEST_CASE("mcp and scad tails: mixed (y, theta) derivatives against differences") {
  for (const auto& c : CASES_S) {
    const std::vector<double> ts(c.begin(), c.end());
    const std::vector<double> tl(c.begin(), c.begin() + 4);
    const double knot = c[2] * c[1];
    for (double u : {0.4 * knot, 1.6 * knot}) {
      const double hy = 1e-3 * std::fmax(1e-3, u);
      check_y_derivs_p<mcp_s>(ts, c[0] + u, hy, 2e-6);
      check_y_derivs_p<scad_s>(ts, c[0] + u, hy, 2e-6);
      check_y_derivs_p<mcp_l>(tl, c[0] + u, hy, 2e-6);
      check_y_derivs_p<scad_l>(tl, c[0] + u, hy, 2e-6);
    }
  }
}

TEST_CASE("mcp and scad tails: the density integrates to one") {
  // The quadrature takes the cuts it is given, and y_kinks() declares the kink at the
  // location and every knot, so the panels never straddle a point where a derivative of
  // the density jumps.
  double worst = 0.0;
  for (const auto& c : CASES_S) {
    const mcp_s d{c[0], c[1], c[2], c[3], c[4]};
    const scad_s e{c[0], c[1], c[2], c[3], c[4]};
    const mcp_l f{c[0], c[1], c[2], c[3]};
    const scad_l g{c[0], c[1], c[2], c[3]};
    auto mass = [&](const auto& law) {
      const auto k = law.y_kinks();
      double cuts[8];
      for (int i = 0; i < k.count; ++i) cuts[i] = k.point[i];
      return laplace::math::quad::panels_quad([&](double y) { return std::exp(law.logpdf(y)); },
                                              -ld::distrib_infinity, ld::distrib_infinity,
                                              law.center(), law.scale(), cuts, k.count)
          .value;
    };
    for (double m : {mass(d), mass(e), mass(f), mass(g)}) {
      INFO("lambda " << c[1] << " p " << c[2] << " mass " << m);
      CHECK(std::fabs(m - 1.0) <= 1e-13);
      worst = std::fmax(worst, std::fabs(m - 1.0));
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad with the Student and logistic tails: worst departure of the mass from one %.3g\n",
                worst);
}

TEST_CASE("mcp and scad tails: the closed cumulative functions") {
  double worst = 0.0;
  for (const auto& c : CASES_S) {
    const mcp_s d{c[0], c[1], c[2], c[3], c[4]};
    const scad_l g{c[0], c[1], c[2], c[3]};
    const double knot = c[2] * c[1];
    // the location is the median
    CHECK(d.cdf(c[0]) == doctest::Approx(0.5).epsilon(1e-14));
    CHECK(g.cdf(c[0]) == doctest::Approx(0.5).epsilon(1e-14));
    for (double u : {0.0, 0.2 * knot, knot, 1.3 * knot, 4.0 * knot + 2.0}) {
      for (double sign : {1.0, -1.0}) {
        const double y = c[0] + sign * u;
        auto one = [&](const auto& law) {
          INFO("y " << y << " cdf " << law.cdf(y));
          CHECK(law.cdf(y) + law.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
          CHECK(law.log_cdf(y) == doctest::Approx(std::log(law.cdf(y))).epsilon(1e-13));
          CHECK(law.log_ccdf(y) == doctest::Approx(std::log(law.ccdf(y))).epsilon(1e-13));
          CHECK(law.cdf(y) >= 0.0);
          CHECK(law.cdf(y) <= 1.0);
          // where the upper probability falls below the resolution of one the cdf is one
          // exactly and the logarithm is what carries the tail
          if (law.cdf(y) == 1.0) CHECK(std::isfinite(law.log_ccdf(y)));
        };
        one(d);
        one(g);
      }
    }
    // the increment of F against the quadrature of the density, over a stretch beyond
    // the last knot, where there is no cut to place
    for (double f : {1.2, 3.0}) {
      const double y1 = c[0] + f * (knot + 1.0), y2 = y1 + 1.5 * (knot + 1.0);
      auto inc = [&](const auto& law) {
        const double q = laplace::math::quad::panels_quad(
                             [&](double t) { return std::exp(law.logpdf(t)); }, y1, y2, law.center(),
                             law.scale(), nullptr, 0)
                             .value;
        INFO("y1 " << y1 << " y2 " << y2 << " increment " << (law.cdf(y2) - law.cdf(y1))
                   << " quadrature " << q);
        CHECK(law.cdf(y2) - law.cdf(y1) == doctest::Approx(q).epsilon(1e-10));
        worst = std::fmax(worst, std::fabs(law.cdf(y2) - law.cdf(y1) - q));
      };
      inc(d);
      inc(g);
    }
    // far out in the tail, where only the logarithms survive: the Student tail is
    // polynomial and the logistic one exponential, so log(1 - F) falls linearly in
    // log u for the first and linearly in u for the second
    const double y1 = c[0] + 40.0 * (knot + 1.0), y2 = c[0] + 80.0 * (knot + 1.0);
    CHECK(d.log_ccdf(y1) < 0.0);
    CHECK(d.log_ccdf(y2) < d.log_ccdf(y1));
    CHECK(std::isfinite(d.log_ccdf(y2)));
    CHECK(std::isfinite(g.log_ccdf(y1)));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("mcp/scad with the Student and logistic tails: worst departure of the increment of F from the quadrature %.3g\n",
                worst);
}

TEST_CASE("mcp and scad tails: the elementary constants of the logistic tail") {
  // T = s and int_x^inf exp(-H) = s (1 - tanh(x/s)) hold at every order, which is what
  // makes this tail the exact reference of the other two
  for (double s : {0.25, 1.0, 3.0, 17.0}) {
    const ld::logistic_tail t{s};
    CHECK(t.tail_const() == s);
    CHECK(t.log_tail_const() == doctest::Approx(std::log(s)).epsilon(1e-15));
    CHECK(t.upper(0.0) == doctest::Approx(s).epsilon(1e-15));
    for (double x : {0.0, 0.3, 1.0, 5.0, 40.0}) {
      const double z = x / s;
      const double want = s * (1.0 - std::tanh(z));
      // log(s (1 - tanh z)) underflows to -infinity in double from z of about 19, which
      // is why log_upper is written as log(2 s) - 2 z - log1p(e^(-2 z)) instead
      const double log_want = 0.69314718055994530942 + std::log(s) - 2.0 * z - std::log1p(std::exp(-2.0 * z));
      INFO("s " << s << " x " << x << " z " << z);
      CHECK(t.log_upper(x) == doctest::Approx(log_want).epsilon(1e-14));
      // 1 - tanh z is about 2 e^(-2 z), so forming it by subtraction keeps its digits
      // only while that is well above eps: at z = 13.3 it is 2.6e-12 and four digits
      // are left, which is why the naive form is compared below z = 3 alone
      if (z <= 3.0) {
        CHECK(t.upper(x) == doctest::Approx(want).epsilon(1e-14));
        CHECK(std::log(want) == doctest::Approx(log_want).epsilon(1e-12));
      }
    }
    // H(0) = H'(0) = 0, which is what keeps the join at the knot C^1
    CHECK(t.psi_deriv(0, 0.0) == 0.0);
    CHECK(t.psi_deriv(1, 0.0) == 0.0);
    CHECK(t.psi_deriv(2, 0.0) == doctest::Approx(2.0).epsilon(1e-15));
  }
  // the Cauchy case of the Student tail: nu = 1 gives T = s pi / 2
  for (double s : {0.25, 1.0, 3.0}) {
    const ld::student_tail t{s, 1.0};
    CHECK(t.tail_const() == doctest::Approx(s * 1.57079632679489661923).epsilon(1e-14));
    CHECK(t.psi_deriv(0, 0.0) == 0.0);
    CHECK(t.psi_deriv(1, 0.0) == 0.0);
    CHECK(t.psi_deriv(2, 0.0) == doctest::Approx(2.0).epsilon(1e-15));
  }
}

TEST_CASE("mcp and scad tails: the score of the three tails far from the location") {
  // What the three tails differ in: beyond the knot the Gaussian score grows without
  // bound, the logistic one tends to 2/s and the Student one to zero, so only the last
  // keeps the behaviour the penalty was built for.
  const double lam = 1.3, p = 3.0, s = 1.0;
  const mcp_g a{0.0, lam, p, s * s};
  const mcp_l b{0.0, lam, p, s};
  const mcp_s c{0.0, lam, p, s, 3.0};
  // measured on 2026-11-19 at s = 1 and nu = 3: the Student score is its closed form
  // (nu + 1) x/(nu s^2 + x^2) to 1.4e-16, so x q' climbs to nu + 1 = 4 (3.8835 at
  // x = 10, 3.99999 at x = 10^4) while the Gaussian score is x itself
  const double nu = 3.0;
  double prev_g = 0.0;
  for (double u : {10.0, 100.0, 1000.0, 10000.0}) {
    const double y = p * lam + u;
    const double qg = -a.logpdf_dy_dk<1, 0>(y)[0];  // q'(u), the score of the penalty
    const double ql = -b.logpdf_dy_dk<1, 0>(y)[0];
    const double qs = -c.logpdf_dy_dk<1, 0>(y)[0];
    INFO("u " << u << " gaussian " << qg << " logistic " << ql << " student " << qs);
    CHECK(qg == doctest::Approx(u / (s * s)).epsilon(1e-13));  // unbounded, x/V
    CHECK(qg > prev_g);
    CHECK(ql == doctest::Approx(2.0 / s).epsilon(1e-6));  // bounded by 2/s
    CHECK(qs == doctest::Approx((nu + 1.0) * u / (nu * s * s + u * u)).epsilon(1e-13));
    CHECK(u * qs < nu + 1.0);
    CHECK(u * qs > (nu + 1.0) * (1.0 - 4.0 * nu * s * s / (u * u)));
    prev_g = qg;
  }
  CHECK(prev_g > 1e3);
}

TEST_CASE("mcp and scad tails: the Gaussian tail is the Student tail's limit") {
  // At a fixed scale the Student exponent tends to x^2/(2 s^2) and T to sqrt(pi s^2/2),
  // both as 1/nu, so the whole log-density does
  const double mu = 0.2, lam = 1.3, p = 3.0, s = 1.1;
  const mcp_g g{mu, lam, p, s * s};
  // measured on 2026-11-19 at y = mu + 9, which is 5.1 beyond the knot: the difference
  // of the log-densities is 9.15e-1, 1.046e-2, 1.047e-4 and 1.049e-6 at nu = 1e2, 1e4,
  // 1e6 and 1e8, so nu times it settles at 104.9 and the difference itself falls like
  // 1/nu
  double prev = 1e300;
  for (double nu : {1e2, 1e4, 1e6, 1e8}) {
    const mcp_s d{mu, lam, p, s, nu};
    const double y = mu + 9.0;  // out in the tail, where the two differ most
    const double diff = std::fabs(d.logpdf(y) - g.logpdf(y));
    INFO("nu " << nu << " difference " << diff << " nu times it " << (nu * diff));
    CHECK(nu * diff <= 110.0);
    CHECK(diff < prev);
    prev = diff;
  }
  CHECK(prev < 2e-6);
  // and the tail's own constant does the same
  for (double nu : {1e4, 1e8}) {
    const ld::student_tail t{s, nu};
    const ld::gaussian_tail u{s * s};
    CHECK(std::fabs(t.tail_const() / u.tail_const() - 1.0) < 3.0 / nu);
  }
}

TEST_CASE("mcp and scad tails: the generic quantile and the generic cdf route") {
  const mcp_s d{0.5, 1.3, 3.0, 1.1, 5.0};
  const scad_l e{0.5, 1.3, 3.7, 1.1};
  for (double pr : {1e-6, 0.01, 0.2, 0.5, 0.75, 0.99, 1.0 - 1e-6}) {
    const double q1 = ld::distrib_quantile(d, pr);
    INFO("p " << pr << " quantile " << q1);
    CHECK(d.cdf(q1) == doctest::Approx(pr).epsilon(1e-10));
    const double q2 = ld::distrib_quantile(e, pr);
    CHECK(e.cdf(q2) == doctest::Approx(pr).epsilon(1e-10));
  }
}

TEST_CASE("mcp and scad tails: start values") {
  ld::distrib_sample_summary s{};
  s.median = 1.5;
  s.mean_abs_dev_median = 0.8;
  s.variance = 2.25;
  const mcp_s a = mcp_s{0.0, 1.0, 3.0, 1.0, 5.0}.start_value(s);
  CHECK(a.mu == doctest::Approx(1.5));
  CHECK(a.lambda > 0.0);
  CHECK(a.gamma > 0.0);
  CHECK(a.tail.tail_scale > 0.0);
  CHECK(a.tail.tail_df > 0.0);
  CHECK(std::isfinite(a.logpdf(1.5)));
  const scad_l b = scad_l{0.0, 1.0, 3.7, 1.0}.start_value(s);
  CHECK(b.a > 1.0);
  CHECK(b.tail.tail_scale > 0.0);
  CHECK(std::isfinite(b.logpdf(1.5)));
}

TEST_CASE("mcp and scad tails: the log-density, the cumulative functions and Z against mpmath") {
  const auto rows = read_reference("distrib_mcp_scad.csv");
  REQUIRE(rows.size() > 1000);
  // the scale of each order at each point, the largest component there
  std::map<std::tuple<int, int, int, double, double, double, double, double>, double> scale;
  for (const auto& r : rows) {
    if (r.kind != 0 || r.K == 0) continue;
    auto& a = scale[std::make_tuple(r.law, r.tail, r.K, r.y, r.lam, r.p, r.t1, r.t2)];
    a = std::fmax(a, std::fabs(r.hi));
  }
  double worst[7] = {0, 0, 0, 0, 0, 0, 0}, worst_cdf = 0.0, worst_z = 0.0;
  std::tuple<double, double, double, double, double, int, int> where[7];
  int checked = 0;
  for (const auto& r : rows) {
    const double ts[5] = {r.mu, r.lam, r.p, r.t1, r.t2};
    auto run = [&](const auto& law) {
      if (r.kind == 3) {
        const double z = std::exp(r.law == 0 ? ld::detail_mcp::log_z(r.lam, r.p, law.tail)
                                             : ld::detail_scad::log_z(r.lam, r.p, law.tail));
        const double err = std::fabs(z - r.hi) / ulp_of(r.hi);
        INFO("Z: law " << r.law << " tail " << r.tail << " lambda " << r.lam << " ulp " << err);
        CHECK(err <= BOUND_Z);
        worst_z = std::fmax(worst_z, err);
        ++checked;
        return;
      }
      if (r.kind == 1 || r.kind == 2) {
        const double v = r.kind == 1 ? law.log_cdf(r.y) : law.log_ccdf(r.y);
        const double err = std::fabs(v - r.hi) / ulp_of(std::fmax(std::fabs(r.hi), 1.0));
        INFO("log F: law " << r.law << " tail " << r.tail << " y " << r.y << " ulp " << err);
        CHECK(err <= BOUND_CDF);
        worst_cdf = std::fmax(worst_cdf, err);
        ++checked;
        return;
      }
      constexpr int P = std::decay_t<decltype(law)>::n_params;
      const auto v = lderiv(law, r.y, r.K);
      const double got = v[r.pos];
      const double sc = r.K == 0 ? std::fmax(std::fabs(r.hi), 1.0)
                                 : scale[std::make_tuple(r.law, r.tail, r.K, r.y, r.lam, r.p, r.t1, r.t2)];
      const double err = std::fabs(got - r.hi) / ulp_of(std::fmax(sc, 1.0));
      if (err > worst[r.K]) {
        worst[r.K] = err;
        where[r.K] = std::make_tuple(r.y, r.lam, r.p, r.t1, r.t2, r.law, r.tail);
      }
      INFO("order " << r.K << " pos " << r.pos << " of " << P << " params, law " << r.law << " tail "
                    << r.tail << " y " << r.y << " ulp " << err);
      CHECK(err <= BOUND_LOGPDF[r.K]);
      ++checked;
    };
    if (r.tail == 1) {
      if (r.law == 0) run(make<mcp_s>(ts));
      else run(make<scad_s>(ts));
    } else {
      if (r.law == 0) run(make<mcp_l>(ts));
      else run(make<scad_l>(ts));
    }
  }
  CHECK(checked > 1000);
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (int K = 0; K <= 6; ++K)
      std::printf("mcp/scad tails order %d: worst %.3g ulp of the order's scale, at y = %g lambda = %g p = %g t1 = %g t2 = %g law %d tail %d\n",
                  K, worst[K], std::get<0>(where[K]), std::get<1>(where[K]), std::get<2>(where[K]),
                  std::get<3>(where[K]), std::get<4>(where[K]), std::get<5>(where[K]),
                  std::get<6>(where[K]));
    std::printf("mcp/scad tails: log F and log(1 - F) worst %.3g ulp of max(|log F|, 1), Z worst %.3g ulp, %d values\n",
                worst_cdf, worst_z, checked);
  }
}

// ---------------------------------------------------------------- the prox in y
// The prox of all three tails is here, the Gaussian one included, because what the
// choice of tail changes is the shrinkage far from the location, which only a comparison
// of the three shows.

struct prox_row {
  int law, tail;
  double lam, p, t1, t2, U, t, hi, lo, gap;
};
constexpr int PROX_FIELDS = 11;

std::vector<prox_row> read_prox_reference() {
  std::vector<prox_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/distrib_mcp_scad_prox.csv";
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if ((int)s.size() != PROX_FIELDS) continue;
    auto num = [&](int i) { return std::strtod(s[i].c_str(), nullptr); };
    prox_row r;
    r.law = std::atoi(s[0].c_str()), r.tail = std::atoi(s[1].c_str());
    r.lam = num(2), r.p = num(3), r.t1 = num(4), r.t2 = num(5), r.U = num(6), r.t = num(7);
    r.hi = num(8), r.lo = num(9), r.gap = num(10);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

// t q(x) + (x - U)^2/2 up to the constant t log Z, which every candidate carries
template <class D>
double prox_objective(const D& d, double x, double U, double t) {
  const double e = x - U;
  return -t * d.logpdf(d.mu + x) + 0.5 * e * e;
}

TEST_CASE("mcp and scad prox: the conventions") {
  for (const auto& c : CASES_S) {
    const double mu = c[0], lam = c[1], p = c[2];
    const mcp_s m{mu, lam, p, c[3], c[4]};
    const scad_s s{mu, lam, p, c[3], c[4]};
    const mcp_l ml{mu, lam, p, c[3]};
    const mcp_g mg{mu, lam, p, c[3]};
    for (const double t : {0.01, 0.5, 1.0, 10.0}) {
      // the location is its own prox, and the sign and the bound hold
      CHECK(m.prox_y(mu, t) == mu);
      CHECK(s.prox_y(mu, t) == mu);
      CHECK(ml.prox_y(mu, t) == mu);
      CHECK(mg.prox_y(mu, t) == mu);
      // an infinite point is its own prox however large the step, and NaN passes through
      const double inf = std::numeric_limits<double>::infinity();
      CHECK(m.prox_y(inf, t) == inf);
      CHECK(s.prox_y(-inf, t) == -inf);
      CHECK(ml.prox_y(inf, t) == inf);
      CHECK(mg.prox_y(-inf, t) == -inf);
      CHECK(std::isnan(m.prox_y(std::numeric_limits<double>::quiet_NaN(), t)));
      CHECK(std::isnan(s.prox_y(std::numeric_limits<double>::quiet_NaN(), t)));
      for (const double U : {1e-6, 0.1, 1.0, 3.0, 50.0, 1e6}) {
        for (int which = 0; which < 4; ++which) {
          const double x = which == 0 ? m.prox_y(mu + U, t) - mu
                                      : (which == 1 ? s.prox_y(mu + U, t) - mu
                                                    : (which == 2 ? ml.prox_y(mu + U, t) - mu
                                                                  : mg.prox_y(mu + U, t) - mu));
          INFO("which " << which << " U " << U << " t " << t);
          CHECK(x >= 0.0);
          CHECK(x <= U * (1.0 + 1e-15));
          // odd about the location: the two prox values sum to twice it, the statement
          // that survives the rounding of mu + x when mu is not zero
          const double lo = which == 0 ? m.prox_y(mu - U, t)
                                       : (which == 1 ? s.prox_y(mu - U, t)
                                                     : (which == 2 ? ml.prox_y(mu - U, t)
                                                                   : mg.prox_y(mu - U, t)));
          const double up = which == 0 ? m.prox_y(mu + U, t)
                                       : (which == 1 ? s.prox_y(mu + U, t)
                                                     : (which == 2 ? ml.prox_y(mu + U, t)
                                                                   : mg.prox_y(mu + U, t)));
          CHECK(std::fabs(lo + up - 2.0 * mu) <= 4.0 * ulp_of(std::fmax(std::fabs(mu), U)));
        }
      }
    }
  }
}

// Where the prox lands strictly inside a piece, the objective's derivative
// t q'(x) + x - U vanishes there, with q'(x) = -d_y log f at mu + x. A named function
// rather than a generic lambda: MSVC ends nvcc's host pass with an internal compiler
// error on a generic lambda instantiated over these laws inside a doctest TEST_CASE,
// which is a static function (met on 2026-11-20; the file already keeps its templated
// bodies in named functions for the related trap of 2026-11-13).
template <class D>
void prox_optimality_one(const D& d, double lam, double p, bool scad, double& worst, int& checked) {
  const double knot = p * lam;
  for (const double t : {0.01, 0.3, 1.0, 4.0})
    for (const double U : {0.05 * knot, 0.6 * knot, 1.3 * knot, 5.0 * knot, 500.0 * knot}) {
      const double x = d.prox_y(U, t);
      if (!(x > 1e-12) || std::fabs(x - knot) < 1e-9 * knot) continue;
      if (scad && std::fabs(x - lam) < 1e-9 * lam) continue;  // at the first knot
      const double qp = -d.template logpdf_dy_dk<1, 0>(x)[0];
      const double g = t * qp + x - U;
      const double sc = std::fmax(std::fabs(t * qp), std::fabs(U));
      INFO("U " << U << " t " << t << " x " << x);
      CHECK(std::fabs(g) <= 1e-12 * std::fmax(sc, 1.0));
      worst = std::fmax(worst, std::fabs(g) / std::fmax(sc, 1.0));
      ++checked;
    }
}

TEST_CASE("mcp and scad prox: first-order optimality where the minimizer is interior") {
  double worst = 0.0;
  int checked = 0;
  for (const auto& c : CASES_S) {
    const double lam = c[1], p = c[2], t1 = c[3], t2 = c[4];
    prox_optimality_one(mcp_g{0.0, lam, p, t1}, lam, p, false, worst, checked);
    prox_optimality_one(mcp_s{0.0, lam, p, t1, t2}, lam, p, false, worst, checked);
    prox_optimality_one(mcp_l{0.0, lam, p, t1}, lam, p, false, worst, checked);
    prox_optimality_one(scad_g{0.0, lam, p, t1}, lam, p, true, worst, checked);
    prox_optimality_one(scad_s{0.0, lam, p, t1, t2}, lam, p, true, worst, checked);
    prox_optimality_one(scad_l{0.0, lam, p, t1}, lam, p, true, worst, checked);
  }
  std::printf("mcp/scad prox: first-order optimality worst %.3g relative, %d points\n", worst, checked);
}

TEST_CASE("mcp and scad prox: the shrinkage far out separates the three tails") {
  // what the properizing tail decides: how much the prox still pulls a point that lies
  // far beyond the last knot. U - x* grows with U for the Gaussian tail, settles at
  // 2t/s for the logistic one and falls to zero for the Student one.
  const double lam = 1.3, p = 3.0, t = 0.7, s = 1.1, nu = 5.0;
  const mcp_g g{0.0, lam, p, s * s};
  const mcp_l l{0.0, lam, p, s};
  const mcp_s st{0.0, lam, p, s, nu};
  double prev_g = 0.0;
  for (const double U : {1e2, 1e3, 1e4, 1e5}) {
    const double sg = U - g.prox_y(U, t), sl = U - l.prox_y(U, t), ss = U - st.prox_y(U, t);
    INFO("U " << U);
    CHECK(sg > prev_g);                       // the Gaussian tail keeps pulling harder
    CHECK(std::fabs(sl - 2.0 * t / s) < 1e-9 * (2.0 * t / s));  // the logistic settles
    CHECK(ss < 2.0 * t / s);                   // the Student pulls less than that
    CHECK(ss * U < 1.5 * t * (nu + 1.0));      // and its pull falls like t(nu + 1)/U
    CHECK(ss * U > 0.5 * t * (nu + 1.0));
    prev_g = sg;
  }
  // beyond the knot the Gaussian tail's shrinkage is exactly (U - K) t/(V + t)
  const double U = 1e4, V = s * s, w = U - p * lam;
  CHECK(std::fabs((U - g.prox_y(U, t)) - w * t / (V + t)) <= 1e-9 * w * t / (V + t));
}

TEST_CASE("mcp and scad prox: the Student tail where the cubic's discriminant cancels") {
  // Beyond w of about 1e11 the discriminant of the stationary cubic is a difference of
  // two terms of size w^6 and its computed sign goes wrong, so the root near w is added
  // by prox_roots on its own account. Here the returned point must still be stationary
  // and must beat the knot, which is what would be returned if no root were found.
  const double lam = 1.3, p = 3.0, s = 1.1, nu = 5.0, kap = 0.5 * (nu + 1.0);
  for (const double t : {0.5, 7.0}) {
    for (const double U : {1e9, 1e11, 1e13, 1e15}) {
      const mcp_s d{0.0, lam, p, s, nu};
      const double x = d.prox_y(U, t), knot = p * lam;
      INFO("U " << U << " t " << t << " x " << x);
      CHECK(x > knot);
      // t q'(x) + x - U = 0, with q' read from the law's own y-derivative
      const double qp = -d.template logpdf_dy_dk<1, 0>(x)[0];
      CHECK(std::fabs(t * qp + x - U) <= 1e-10 * U);
      // and its objective is below the knot's
      CHECK(prox_objective(d, x, U, t) < prox_objective(d, knot, U, t));
      // the pull is t (nu + 1) / w, so the point is all but U itself
      CHECK(U - x < 4.0 * t * kap / (U - knot) + 1e-9 * U);
    }
  }
}

// one reference row against the law it names; see prox_optimality_one on why this is a
// named function and not a lambda
struct prox_ref_state {
  double worst = 0.0, worst_obj = 0.0;
  int checked = 0, flat = 0;
  std::tuple<double, double, double, int, int> where{};
};

template <class D>
void prox_reference_one(const D& d, const prox_row& r, prox_ref_state& st) {
  const double x = d.prox_y(r.U, r.t), ref = r.hi + r.lo;
  // the objective at the computed point never exceeds the reference's: this holds
  // whether or not the two agree on which branch wins
  const double go = prox_objective(d, x, r.U, r.t), gr = prox_objective(d, ref, r.U, r.t);
  const double sc = std::fmax(std::fabs(gr), 1.0);
  INFO("law " << r.law << " tail " << r.tail << " lam " << r.lam << " p " << r.p << " U " << r.U
              << " t " << r.t << " x " << x << " ref " << ref << " gap " << r.gap);
  CHECK(go - gr <= 64.0 * ulp_of(sc));
  st.worst_obj = std::fmax(st.worst_obj, (go - gr) / sc);
  if (r.gap == 0.0) {  // the objective is flat: every point of a piece minimizes
    ++st.flat;
    return;
  }
  const double e = std::fabs(x - ref) / ulp_of(std::fmax(std::fabs(ref), 1e-300));
  CHECK(e <= BOUND_PROX);
  if (e > st.worst) st.worst = e, st.where = {r.U, r.t, r.lam, r.law, r.tail};
  ++st.checked;
}

void prox_reference_row(const prox_row& r, prox_ref_state& st) {
  if (r.law == 0) {
    if (r.tail == 0) prox_reference_one(mcp_g{0.0, r.lam, r.p, r.t1}, r, st);
    else if (r.tail == 1) prox_reference_one(mcp_s{0.0, r.lam, r.p, r.t1, r.t2}, r, st);
    else prox_reference_one(mcp_l{0.0, r.lam, r.p, r.t1}, r, st);
  } else {
    if (r.tail == 0) prox_reference_one(scad_g{0.0, r.lam, r.p, r.t1}, r, st);
    else if (r.tail == 1) prox_reference_one(scad_s{0.0, r.lam, r.p, r.t1, r.t2}, r, st);
    else prox_reference_one(scad_l{0.0, r.lam, r.p, r.t1}, r, st);
  }
}

TEST_CASE("mcp and scad prox: against mpmath") {
  const auto rows = read_prox_reference();
  REQUIRE(rows.size() > 10000);
  prox_ref_state st;
  for (const auto& r : rows) prox_reference_row(r, st);
  const double worst = st.worst, worst_obj = st.worst_obj;
  const int checked = st.checked, flat = st.flat;
  const auto where = st.where;
  std::printf("mcp/scad prox: worst %.3g ulp at U = %g t = %g lambda = %g law %d tail %d; "
              "objective never above the reference's by more than %.3g relative; %d values, %d flat\n",
              worst, std::get<0>(where), std::get<1>(where), std::get<2>(where), std::get<3>(where),
              std::get<4>(where), worst_obj, checked, flat);
}

// --------------------------------------------------------------- the kernels
// An extended device lambda must not live in a function with internal linkage, which a
// doctest TEST_CASE body is, so every kernel is launched from a named function with
// external linkage (see the trap of 2026-11-13 in CLAUDE.md).

template <class D>
LAPLACE_HD D build(double t0, double t1, double t2, double t3, double t4) {
  if constexpr (D::n_params == 5) return D{t0, t1, t2, t3, t4};
  else return D{t0, t1, t2, t3};
}

template <class D, int K>
void tails_kernel_order(const double* th, double bound) {
  constexpr int P = D::n_params;
  const double y = th[0] + 2.4 * th[1] * th[2] + 0.7;  // out in the tail
  const auto expect = as_vec<P, K>(build<D>(th[0], th[1], th[2], th[3], th[4]).template logpdf_dk<K>(y));
  Kokkos::View<double*> out("out", ld::sym_tensor<P, K>::size);
  const double t0 = th[0], t1 = th[1], t2 = th[2], t3 = th[3], t4 = th[4];
  Kokkos::parallel_for(
      "mcp_scad_tails", 1, KOKKOS_LAMBDA(int) {
        const auto t = build<D>(t0, t1, t2, t3, t4).template logpdf_dk<K>(y);
        for (int i = 0; i < ld::sym_tensor<P, K>::size; ++i) out(i) = t[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0, sc = 0.0;
  for (int i = 0; i < (int)expect.size(); ++i) sc = std::fmax(sc, std::fabs(expect[i]));
  for (int i = 0; i < (int)expect.size(); ++i) {
    INFO("order " << K << " component " << i);
    CHECK(std::fabs(h(i) - expect[i]) <= bound * std::fmax(sc, 1.0));
    worst = std::fmax(worst, std::fabs(h(i) - expect[i]) / std::fmax(sc, 1.0));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS") && worst > 0.0)
    std::printf("mcp/scad tails kernel order %d: worst host-device difference %.3g of the order's scale\n",
                K, worst);
}

template <class D>
void tails_kernel_other(const double* th) {
  const double knot = th[1] * th[2];
  Kokkos::View<double*> out("out", 6);
  const double t0 = th[0], t1 = th[1], t2 = th[2], t3 = th[3], t4 = th[4];
  Kokkos::parallel_for(
      "mcp_scad_tails_other", 1, KOKKOS_LAMBDA(int) {
        const D d = build<D>(t0, t1, t2, t3, t4);
        const double y = t0 + 1.8 * knot;
        out(0) = d.logpdf(y);
        out(1) = d.cdf(y);
        out(2) = d.ccdf(y);
        out(3) = d.log_cdf(y);
        out(4) = d.log_ccdf(y);
        out(5) = double(d.y_kinks().count);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  const auto d = build<D>(th[0], th[1], th[2], th[3], th[4]);
  const double y = th[0] + 1.8 * knot;
  CHECK(h(0) == doctest::Approx(d.logpdf(y)).epsilon(1e-13));
  CHECK(h(1) == doctest::Approx(d.cdf(y)).epsilon(1e-12));
  CHECK(h(2) == doctest::Approx(d.ccdf(y)).epsilon(1e-12));
  CHECK(h(3) == doctest::Approx(d.log_cdf(y)).epsilon(1e-12));
  CHECK(h(4) == doctest::Approx(d.log_ccdf(y)).epsilon(1e-12));
  CHECK(h(5) == double(d.y_kinks().count));
}

void tails_kernel_case() {
  const double ts[5] = {0.4, 1.3, 3.0, 1.1, 5.0};
  const double tl[5] = {0.4, 1.3, 3.0, 1.1, 1.0};
  tails_kernel_order<mcp_s, 0>(ts, 1e-14);
  tails_kernel_order<mcp_s, 1>(ts, 2e-15);
  tails_kernel_order<mcp_s, 2>(ts, 2e-15);
  tails_kernel_order<mcp_s, 3>(ts, 2e-15);
  tails_kernel_order<mcp_s, 4>(ts, 2e-15);
  tails_kernel_order<mcp_s, 5>(ts, 2e-15);
  tails_kernel_order<mcp_s, 6>(ts, 2e-15);
  tails_kernel_order<scad_s, 6>(ts, 2e-15);
  tails_kernel_order<mcp_l, 6>(tl, 2e-15);
  tails_kernel_order<scad_l, 6>(tl, 2e-15);
  tails_kernel_other<mcp_s>(ts);
  tails_kernel_other<scad_s>(ts);
  tails_kernel_other<mcp_l>(tl);
  tails_kernel_other<scad_l>(tl);
}

TEST_CASE("mcp and scad tails: every order inside a kernel") { tails_kernel_case(); }

// the prox inside a kernel: the device must reach the same branch, which for the Student
// tail means the same root of the cubic
template <class D>
void prox_kernel_one(const double* th, int n_points) {
  const D d = build<D>(th[0], th[1], th[2], th[3], th[4]);
  const double t = 0.7;
  std::vector<double> expect(n_points);
  for (int i = 0; i < n_points; ++i)
    expect[i] = d.prox_y(th[0] + std::pow(2.0, double(i - n_points / 2)) * th[1] * th[2], t);
  Kokkos::View<double*> out("prox", n_points);
  const double t0 = th[0], t1 = th[1], t2 = th[2], t3 = th[3], t4 = th[4];
  const int np = n_points;
  Kokkos::parallel_for(
      "mcp_scad_prox", n_points, KOKKOS_LAMBDA(int i) {
        const auto dd = build<D>(t0, t1, t2, t3, t4);
        out(i) = dd.prox_y(t0 + Kokkos::pow(2.0, double(i - np / 2)) * t1 * t2, t);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n_points; ++i) {
    INFO("point " << i << " host " << expect[i] << " device " << h(i));
    const double sc = std::fmax(std::fabs(expect[i]), 1.0);
    CHECK(std::fabs(h(i) - expect[i]) <= 1e-13 * sc);
    worst = std::fmax(worst, std::fabs(h(i) - expect[i]) / sc);
  }
  std::printf("mcp/scad prox in a kernel: host against device worst %.3g relative over %d points\n",
              worst, n_points);
}

void prox_kernel_case() {
  const double ts[5] = {0.4, 1.3, 3.0, 1.1, 5.0};
  const double tl[5] = {0.4, 1.3, 3.0, 1.1, 1.0};
  prox_kernel_one<mcp_s>(ts, 24);
  prox_kernel_one<scad_s>(ts, 24);
  prox_kernel_one<mcp_l>(tl, 24);
  prox_kernel_one<scad_g>(tl, 24);
}

TEST_CASE("mcp and scad prox: inside a kernel") { prox_kernel_case(); }
