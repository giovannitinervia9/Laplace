// beta_distrib (mean mu, precision phi; include/laplace/distrib/beta.hpp): descriptors, every derivative
// order against a 5-point difference of the order below (in the parameters, in y and mixed), the facts that
// follow from the data entering linearly in the shapes (the derivatives of order 3 and more do not depend on
// y; the observed mixed second derivative is the expected one plus the residual R; I_mu,phi vanishes at
// mu = 1/2; the symmetry (mu, y) -> (1 - mu, 1 - y)), the expected information against the variance of the
// score and against the generic quadrature route in both forms, its derivatives against differences, the
// derivatives of log F and log(1 - F) against differences and against the generic quadrature route, the
// cdf family, the quantile in both tails, the draw, the starting value, a kernel launch, and the references
// of tools/reference/distrib_beta.py (mpmath, the definitions): the log-density and its derivatives in
// (mu, phi) to order 6 over phi from 1e-2 to 1e8, the information, the cdf family and the derivatives of
// log F and log(1 - F).
//
// The bounds of the reference tests are PROVISIONAL: they were set from an emulation of the assembly in IEEE
// doubles with correctly rounded polygamma inputs (plus inputs perturbed by 3 ulp) and not from a run of this
// file; the first run on the CPU and on the CUDA build replaces them by one and a half times the measured
// worst error. The two corners where the straightforward assembly loses digits are tested with their own
// bounds and named as such: the pure phi components, which carry no factor of phi since they are built
// from the polygamma remainders, and, scaled by the order's largest component, everything else.
#include <doctest/doctest.h>

#include <laplace/distrib/beta.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/math/special.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included
using laplace_test::hide;

namespace {

using bd = ld::beta_distrib;

constexpr double eps = 2.220446049250313e-16;

// a point (mu, phi) of the tests
struct point {
  double mu, phi;
};

// the generic route returns the closed-form container for K = 0 and the derivative container for K >= 1
template <int K, class T>
std::vector<double> generic_info_vec(const T& t) {
  if constexpr (K == 0) return as_vec<2, 2>(t);
  else return info_vec<2, K>(t);
}

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

// the data lines of a reference file, split at the commas; the first line (the header) is dropped
std::vector<std::vector<std::string>> read_rows(const char* name, size_t nfields) {
  std::vector<std::vector<std::string>> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    auto s = split_csv(line);
    if (s.size() == nfields) rows.push_back(std::move(s));
  }
  std::fclose(f);
  return rows;
}

double num(const std::string& s) { return std::strtod(s.c_str(), nullptr); }
int integer(const std::string& s) { return std::atoi(s.c_str()); }

// ulp of v (the spacing of the doubles at |v|)
double ulp_at(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

}  // namespace

TEST_CASE("distrib beta: descriptors") {
  CHECK(bd::kind == ld::distrib_kind::continuous);
  CHECK(bd::n_params == 2);
  CHECK(bd::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(bd::params[j].regularity == ld::infinitely_smooth);
    CHECK(bd::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(bd::params[j].moves_support == false);
    CHECK(bd::params[j].circular == false);
  }
  CHECK(bd::params[0].lower == 0.0);
  CHECK(bd::params[0].upper == 1.0);
  CHECK(bd::params[1].lower == 0.0);
  CHECK(bd::params[1].upper == ld::distrib_infinity);
  const bd d{0.25, 7.0};
  CHECK(d.param_kinks(0, 0.3).count == 0);
  CHECK(d.param_kinks(1, 0.3).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == 1.0);
  CHECK(d.center() == 0.25);
  CHECK(d.scale() == doctest::Approx(std::sqrt(0.25 * 0.75 / 8.0)).epsilon(1e-15));
  CHECK(d.logpdf(0.0) == -ld::distrib_infinity);
  CHECK(d.logpdf(1.0) == -ld::distrib_infinity);
  CHECK(d.logpdf(-0.5) == -ld::distrib_infinity);
  CHECK(d.cdf(0.0) == 0.0);
  CHECK(d.cdf(1.0) == 1.0);
  CHECK(d.ccdf(0.0) == 1.0);
  CHECK(d.ccdf(1.0) == 0.0);
  CHECK(d.log_cdf(0.0) == -ld::distrib_infinity);
  CHECK(d.log_cdf(1.0) == 0.0);
  CHECK(d.log_ccdf(0.0) == 0.0);
  CHECK(d.log_ccdf(1.0) == -ld::distrib_infinity);
}

TEST_CASE("distrib beta: the value against the Beta(alpha, beta) density") {
  // l = -log B(alpha, beta) + (alpha - 1) log y + (beta - 1) log(1 - y), alpha = mu phi, beta = (1 - mu) phi
  for (point p : {point{0.3, 6.0}, point{0.5, 2.0}, point{0.8, 10.0}}) {
    const bd d{p.mu, p.phi};
    const double a = p.mu * p.phi, b = (1.0 - p.mu) * p.phi;
    for (double y : {0.05, 0.3, 0.6, 0.95}) {
      const double ref = std::lgamma(p.phi) - std::lgamma(a) - std::lgamma(b) + (a - 1.0) * std::log(y) +
                         (b - 1.0) * std::log1p(-y);
      CHECK(d.logpdf(y) == doctest::Approx(ref).epsilon(1e-13));
    }
  }
  // Beta(1, 1) is the uniform law: mu = 1/2, phi = 2
  CHECK(bd{0.5, 2.0}.logpdf(0.37) == doctest::Approx(0.0).epsilon(1e-14).scale(1.0));
}

TEST_CASE("distrib beta: derivatives against differences") {
  for (double y : {0.02, 0.3, 0.55, 0.9}) check_derivs<bd>({0.3, 2.5}, y, 1e-7);
  for (double y : {0.1, 0.7, 0.95}) check_derivs<bd>({0.7, 0.4}, y, 1e-7);
  for (double y : {0.2, 0.5, 0.8}) check_derivs<bd>({0.5, 7.0}, y, 1e-7);
  for (double y : {0.2, 0.4, 0.6}) check_derivs<bd>({0.4, 200.0}, y, 1e-6);
  for (double y : {0.01, 0.2, 0.9}) check_derivs<bd>({0.02, 3.0}, y, 1e-6);  // alpha = 0.06
}

TEST_CASE("distrib beta: derivatives in y and mixed (y, theta) against differences") {
  const bd d{0.3, 2.5};
  CHECK(d.logpdf_dy_dk<0, 0>(0.7)[0] == doctest::Approx(d.logpdf(0.7)).epsilon(1e-15));
  CHECK(d.logpdf_dy_dk<0, 3>(0.7)[2] == d.logpdf_d3(0.7)[2]);
  // the score in y by hand: (alpha - 1)/y - (beta - 1)/(1 - y)
  const double alpha = 0.3 * 2.5, beta = 0.7 * 2.5;
  CHECK(d.logpdf_dy_dk<1, 0>(0.37)[0] ==
        doctest::Approx((alpha - 1.0) / 0.37 - (beta - 1.0) / 0.63).epsilon(1e-14));
  CHECK(d.logpdf_dy_dk<2, 0>(0.37)[0] ==
        doctest::Approx(-(alpha - 1.0) / (0.37 * 0.37) - (beta - 1.0) / (0.63 * 0.63)).epsilon(1e-14));
  for (double y : {0.05, 0.37, 0.9}) check_y_derivs<bd>({0.3, 2.5}, y, 1e-3 * std::fmin(y, 1.0 - y), 1e-7);
  for (double y : {0.2, 0.6}) check_y_derivs<bd>({0.8, 10.0}, y, 1e-3 * std::fmin(y, 1.0 - y), 1e-7);
}

TEST_CASE("distrib beta: the derivatives of order 3 and more do not depend on y, bit for bit") {
  for (point p : {point{0.3, 2.5}, point{0.5, 7.0}, point{0.02, 3.0}, point{0.9, 1000.0}}) {
    const bd d{p.mu, p.phi};
    for (double y2 : {0.0001, 0.37, 0.999}) {
      const auto a3 = d.logpdf_d3(0.2), b3 = d.logpdf_d3(y2);
      for (int i = 0; i < 4; ++i) CHECK(a3[i] == b3[i]);
      const auto a4 = d.logpdf_d4(0.2), b4 = d.logpdf_d4(y2);
      for (int i = 0; i < 5; ++i) CHECK(a4[i] == b4[i]);
      const auto a5 = d.logpdf_d5(0.2), b5 = d.logpdf_d5(y2);
      for (int i = 0; i < 6; ++i) CHECK(a5[i] == b5[i]);
      const auto a6 = d.logpdf_d6(0.2), b6 = d.logpdf_d6(y2);
      for (int i = 0; i < 7; ++i) CHECK(a6[i] == b6[i]);
    }
    // and the y terms are the whole of the dependence on y at orders 1 and 2: the order-2 components other
    // than the mixed one do not depend on y either
    const auto s2a = d.logpdf_d2(0.2), s2b = d.logpdf_d2(0.7);
    CHECK(s2a[0] == s2b[0]);
    CHECK(s2a[2] == s2b[2]);
  }
}

TEST_CASE("distrib beta: the observed mixed second derivative is the expected one plus the residual R") {
  // l_mu,phi = R - I_mu,phi, R = log(y/(1 - y)) - psi(alpha) + psi(beta), E[R] = 0
  for (point p : {point{0.3, 2.5}, point{0.5, 7.0}, point{0.8, 30.0}, point{0.1, 0.5}, point{0.6, 400.0}}) {
    const bd d{p.mu, p.phi};
    const double nu = 1.0 - p.mu, a = p.mu * p.phi, b = nu * p.phi;
    const auto info = d.expected_info();
    for (double y : {0.05, 0.4, 0.75, 0.97}) {
      const double R = (std::log(y) - std::log1p(-y)) - laplace::math::digamma(a) + laplace::math::digamma(b);
      const double l12 = d.logpdf_d2(y)[1];
      // the rounding of the two sides is that of the terms they are made of
      const double terms = std::fabs(p.mu * p.phi * laplace::math::trigamma(a)) +
                           std::fabs(nu * p.phi * laplace::math::trigamma(b)) + std::fabs(R) +
                           std::fabs(laplace::math::digamma(a)) + std::fabs(laplace::math::digamma(b)) + 1.0;
      INFO("mu = " << p.mu << ", phi = " << p.phi << ", y = " << y);
      CHECK(std::fabs((l12 + info[1]) - R) <= 64.0 * eps * terms);
    }
  }
}

TEST_CASE("distrib beta: I_mu,phi vanishes at mu = 1/2, and the symmetry of the family") {
  for (double phi : {0.05, 0.7, 2.0, 15.0, 1e3, 1e6}) CHECK(bd{0.5, phi}.expected_info()[1] == 0.0);
  // (mu, y) -> (1 - mu, 1 - y): l^(i,j)(mu, y) = (-1)^i l^(i,j)(1 - mu, 1 - y); i derivatives in mu
  for (point p : {point{0.25, 2.5}, point{0.125, 40.0}, point{0.375, 0.6}}) {
    const bd d1{p.mu, p.phi}, d2{1.0 - p.mu, p.phi};
    for (double y : {0.125, 0.375, 0.5, 0.875}) {
      INFO("mu = " << p.mu << ", phi = " << p.phi << ", y = " << y);
      CHECK(d1.logpdf(y) == doctest::Approx(d2.logpdf(1.0 - y)).epsilon(1e-13));
      for (int K = 1; K <= 6; ++K) {
        const auto t1 = lderiv(d1, y, K), t2 = lderiv(d2, 1.0 - y, K);
        double scale = 0.0;
        for (double e : t1) scale = std::fmax(scale, std::fabs(e));
        for (int b = 0; b <= K; ++b) {
          const double sgn = ((K - b) % 2 == 0) ? 1.0 : -1.0;
          CHECK(std::fabs(t1[b] - sgn * t2[b]) <= 1e-12 * scale);
        }
      }
    }
  }
}

TEST_CASE("distrib beta: expected information against the variance of the score") {
  // shapes of at least 1.5 so that the density has no singularity at the ends for the quadrature
  for (point p : {point{0.3, 6.0}, point{0.5, 3.0}, point{0.8, 10.0}, point{0.6, 60.0}}) {
    const bd d{p.mu, p.phi};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("mu = " << p.mu << ", phi = " << p.phi);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-7 * e[0]);
    CHECK(std::fabs(q[1] - e[1]) <= 1e-7 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-7 * e[2]);
  }
}

TEST_CASE("distrib beta: expected information by the generic quadrature route, in both forms") {
  for (point p : {point{0.3, 6.0}, point{0.5, 3.0}}) {
    const bd d{p.mu, p.phi};
    const auto h = hide(d);
    const auto ex0 = as_vec<2, 2>(d.expected_info());
    const double s0 = std::fmax(std::fabs(ex0[0]), std::fabs(ex0[2]));
    const auto sc0 = generic_info_vec<0>(ld::expected_info_by_integral<0, ld::info_form::score>(h));
    const auto he0 = generic_info_vec<0>(ld::expected_info_by_integral<0, ld::info_form::hessian>(h));
    INFO("mu = " << p.mu << ", phi = " << p.phi);
    for (int i = 0; i < 3; ++i) {
      CHECK(std::fabs(sc0[i] - ex0[i]) <= 1e-7 * s0);
      CHECK(std::fabs(he0[i] - ex0[i]) <= 1e-7 * s0);
    }
    const auto ex1 = info_vec<2, 1>(d.expected_info_d1());
    double s1 = 0.0;
    for (double v : ex1) s1 = std::fmax(s1, std::fabs(v));
    const auto sc1 = generic_info_vec<1>(ld::expected_info_by_integral<1, ld::info_form::score>(h));
    const auto he1 = generic_info_vec<1>(ld::expected_info_by_integral<1, ld::info_form::hessian>(h));
    for (int i = 0; i < (int)ex1.size(); ++i) {
      CHECK(std::fabs(sc1[i] - ex1[i]) <= 1e-6 * s1);
      CHECK(std::fabs(he1[i] - ex1[i]) <= 1e-6 * s1);
    }
  }
}

TEST_CASE("distrib beta: derivatives of the expected information against differences") {
  check_info_derivs<bd>({0.3, 2.5}, 1e-7);
  check_info_derivs<bd>({0.7, 0.4}, 1e-7);
  check_info_derivs<bd>({0.5, 7.0}, 1e-7);
  check_info_derivs<bd>({0.4, 200.0}, 1e-6);
  check_info_derivs<bd>({0.02, 3.0}, 1e-6);
}

TEST_CASE("distrib beta: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {0.05, 0.3, 0.7}) check_cdf_derivs<bd>({0.3, 3.0}, y, upper, 1e-6);
    for (double y : {0.1, 0.5, 0.9}) check_cdf_derivs<bd>({0.5, 7.0}, y, upper, 1e-6);
    for (double y : {0.02, 0.3}) check_cdf_derivs<bd>({0.1, 0.8}, y, upper, 1e-6);
  }
}

TEST_CASE("distrib beta: derivatives of log F and log(1 - F), closed route against the generic quadrature") {
  // the closed route (the shape derivatives of the incomplete beta composed with the linear maps) against the
  // quadrature of f B_T(l) of generic_cdf.hpp, every order: the comparison that justifies the closed route
  // (0.1, 10) is here because the mpmath reference and the closed route
  // disagree at it by 21.5 absolute at order six, growing about twentyfold per
  // order and worst in the pure-mu component. The quadrature is the third
  // route that says which of the two is wrong, and it is an independent one:
  // it reads only logpdf and logpdf_dk. It agrees with mpmath, so the closed
  // route is the one at fault.
  //
  // The cause is known and lies below this family, in the route selection of
  // `math/incomplete_beta_deriv.hpp`, not in the composition here (which is
  // exact algebra over two linear shape maps and holds at the two points
  // above). That module chooses which side of the incomplete beta to compute
  // from the sign of lambda = a - (a + b) x. Here alpha = mu phi is exactly 1
  // and x = y = 0.1, so a/(a + b) equals x and lambda is zero in exact
  // arithmetic; in double 9 * 0.1 is not 0.9, lambda comes out at -5.55e-17,
  // and the negative branch computes the opposite side and recovers the
  // requested one by a complement composition, which is the step that loses
  // the digits. Computing the requested side directly gives 1.9e-16 at order
  // one and 1.1e-13 at order six. The region is bounded but wider than the
  // single point: at (1, 9, 0.16) and (1, 30, 0.0516), where lambda is -0.6,
  // the public route gives 5.5e-9 and 1.4e-6 at orders one and six.
  //
  // This case is left FAILING on purpose, by agreement with the session that
  // owns that module, until the route selection is repaired there. A bound of
  // its own would hide a defect that a user fitting a beta with mu phi near
  // one would meet.
  for (point p : {point{0.3, 3.0}, point{0.5, 7.0}, point{0.1, 10.0}}) {
    const bd d{p.mu, p.phi};
    const auto h = hide(d);
    for (double y : {0.1, 0.3, 0.6}) {
      INFO("mu = " << p.mu << ", phi = " << p.phi << ", y = " << y);
#define BETA_CMP(K)                                                                                   \
  {                                                                                                   \
    const auto c = as_vec<2, K>(d.log_cdf_d##K(y)), g = as_vec<2, K>(ld::log_cdf_dk_by_integral<K>(h, y)); \
    const auto cu = as_vec<2, K>(d.log_ccdf_d##K(y)), gu = as_vec<2, K>(ld::log_ccdf_dk_by_integral<K>(h, y)); \
    double s = 0.0, su = 0.0;                                                                         \
    for (double v : c) s = std::fmax(s, std::fabs(v));                                                \
    for (double v : cu) su = std::fmax(su, std::fabs(v));                                             \
    for (int i = 0; i <= K; ++i) {                                                                    \
      CHECK(std::fabs(c[i] - g[i]) <= 1e-6 * s);                                                      \
      CHECK(std::fabs(cu[i] - gu[i]) <= 1e-6 * su);                                                   \
    }                                                                                                 \
  }
      BETA_CMP(1)
      BETA_CMP(2)
      BETA_CMP(3)
      BETA_CMP(4)
      BETA_CMP(5)
      BETA_CMP(6)
#undef BETA_CMP
    }
  }
}

TEST_CASE("distrib beta: cdf, ccdf, logarithms") {
  for (point p : {point{0.3, 6.0}, point{0.5, 2.0}, point{0.9, 40.0}, point{0.02, 3.0}, point{0.6, 400.0}}) {
    const bd d{p.mu, p.phi};
    for (double y : {0.001, 0.05, 0.3, 0.5, 0.8, 0.99}) {
      INFO("mu = " << p.mu << ", phi = " << p.phi << ", y = " << y);
      CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-13));
      if (d.cdf(y) > 1e-300 && d.ccdf(y) > 1e-300) {
        CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-10));
        CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-10));
      }
    }
  }
  // Beta(1, 1) is the uniform law
  const bd u{0.5, 2.0};
  for (double y : {0.1, 0.5, 0.9}) {
    CHECK(u.cdf(y) == doctest::Approx(y).epsilon(1e-14));
    CHECK(u.ccdf(y) == doctest::Approx(1.0 - y).epsilon(1e-14));
  }
}

TEST_CASE("distrib beta: the quantile inverts the cdf, also from log p and log(1 - p) in the tails") {
  for (point p : {point{0.3, 6.0}, point{0.5, 2.0}, point{0.9, 40.0}, point{0.6, 400.0}}) {
    const bd d{p.mu, p.phi};
    for (double q : {1e-10, 1e-3, 0.3, 0.5, 0.9, 1.0 - 1e-6}) {
      INFO("mu = " << p.mu << ", phi = " << p.phi << ", p = " << q);
      CHECK(d.cdf(d.quantile(q)) == doctest::Approx(q).epsilon(1e-9));
    }
  }
  // the log forms, down to where y can still resolve the tail: log p = alpha log y at the lower end (y of the
  // order of 1e-265 at log p = -550), and log q = beta log(1 - y) at the upper one, where y within 2^-53 of 1 is
  // the limit
  const bd d{0.3, 3.0};
  const double alpha = 0.9, beta = 2.1;
  for (double lp : {-0.01, -5.0, -50.0, -300.0, -550.0}) {
    const double x = d.quantile_from_log(lp);
    INFO("log p = " << lp << ", x = " << x);
    CHECK(x > 0.0);
    CHECK(std::fabs(d.log_cdf(x) - lp) <= 1e-9 * std::fabs(lp) + 4.0 * alpha * eps);
  }
  for (double lq : {-0.01, -5.0, -30.0, -60.0}) {
    const double x = d.quantile_from_log_upper(lq);
    INFO("log q = " << lq << ", x = " << x);
    CHECK(x < 1.0);
    // the rounding of x moves log(1 - x) by beta eps / (1 - x)
    CHECK(std::fabs(d.log_ccdf(x) - lq) <= 1e-9 * std::fabs(lq) + 4.0 * beta * eps / (1.0 - x));
  }
}

TEST_CASE("distrib beta: the draw by inversion") {
  const bd d{0.3, 8.0};
  double s = 0.0;
  const int n = 4000;
  for (int i = 0; i < n; ++i) {
    const double y = ld::rng_by_inversion(d, 20261115u, static_cast<std::uint64_t>(i));
    CHECK(y > 0.0);
    CHECK(y < 1.0);
    CHECK(y == ld::rng_by_inversion(d, 20261115u, static_cast<std::uint64_t>(i)));  // the same index, the same draw
    s += y;
  }
  // the mean of n draws has standard deviation scale()/sqrt(n)
  CHECK(std::fabs(s / n - 0.3) <= 6.0 * d.scale() / std::sqrt(static_cast<double>(n)));
}

TEST_CASE("distrib beta: starting value by moments") {
  const bd proto{0.5, 1.0};
  // a sample with mean 0.45 and variance of the sample
  const std::vector<double> y = {0.2, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.25};
  const auto s = ld::summarize_sample(y);
  const bd t = proto.start_value(s);
  CHECK(t.mu == doctest::Approx(s.mean).epsilon(1e-15));
  CHECK(t.phi == doctest::Approx(s.mean * (1.0 - s.mean) / s.variance - 1.0).epsilon(1e-14));
  // no spread: no moment estimate of phi, phi = 1
  const auto s0 = ld::summarize_sample(std::vector<double>(5, 0.3));
  const bd t0 = proto.start_value(s0);
  CHECK(t0.mu == doctest::Approx(0.3).epsilon(1e-15));
  CHECK(t0.phi == 1.0);
  // a mean outside (0, 1): mu clamped inside, phi = 1
  const auto s1 = ld::summarize_sample(std::vector<double>{2.0, 3.0, 4.0});
  const bd t1 = proto.start_value(s1);
  CHECK(t1.mu == doctest::Approx(1.0 - 1e-6).epsilon(1e-15));
  CHECK(t1.phi == 1.0);
  // a mean at the lower clamp
  const auto s2 = ld::summarize_sample(std::vector<double>{0.0, 0.0, 1e-9});
  const bd t2 = proto.start_value(s2);
  CHECK(t2.mu == 1e-6);
  CHECK(t2.phi > 0.0);
  // a variance so large that the moment estimate is below 1e-3: the sample {0, 1, 0, 1} has variance mu (1 - mu)
  const auto s3 = ld::summarize_sample(std::vector<double>{0.0, 1.0, 0.0, 1.0});
  CHECK(proto.start_value(s3).phi == 1e-3);
  CHECK(proto.start_value(s3).mu == doctest::Approx(0.5).epsilon(1e-15));
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp for why (doctest gives each case an internal-linkage
// function whose number restarts per translation unit, and nvcc tags every extended lambda with that address).
void beta_kernel_case() {
  const int n = 64;
  Kokkos::View<double**> six("six", n, 7);
  Kokkos::View<double*> rest("rest", n);
  Kokkos::parallel_for(
      "distrib_beta", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.1 + 0.8 * i / n;
        const ld::beta_distrib g{0.3, 6.0};
        const auto t = g.logpdf_d6(y);
        for (int b = 0; b <= 6; ++b) six(i, b) = t[b];
        rest(i) = g.logpdf(y) + g.logpdf_d1(y)[1] + g.logpdf_dy_dk<3, 2>(y)[1] + g.expected_info_d4().pair[1][2] +
                  g.expected_info_d4().pair[2][4] + g.log_cdf_d6(y)[3] + g.log_ccdf_d6(y)[1] + g.log_cdf(y) +
                  g.log_ccdf(y) + g.quantile(0.3 + 0.4 * i / n);
      });
  auto h6 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), six);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rest);
  for (int i = 0; i < n; ++i) {
    const double y = 0.1 + 0.8 * i / n;
    const ld::beta_distrib g{0.3, 6.0};
    const auto t = g.logpdf_d6(y);
    // Host and device run the same code and may differ in the last bit, because
    // nvcc contracts a * b + c into an FMA where the host does not. The
    // tolerance is relative to the component with a floor at the order's own
    // scale: a component that is small against the largest of its tensor is a
    // difference of larger terms, so its absolute error is set by those terms
    // and not by itself. Measured on CUDA: 1.67e-16 on a component of 1.12e-2,
    // which is 1.5e-14 relative and was just over a bare 1.3e-14.
    double big = 0.0;
    for (int b = 0; b <= 6; ++b) big = std::fmax(big, std::fabs(t[b]));
    for (int b = 0; b <= 6; ++b)
      CHECK(std::fabs(h6(i, b) - t[b]) <= 1.3e-14 * std::fabs(t[b]) + 16.0 * eps * big);
    const double r = g.logpdf(y) + g.logpdf_d1(y)[1] + g.logpdf_dy_dk<3, 2>(y)[1] + g.expected_info_d4().pair[1][2] +
                     g.expected_info_d4().pair[2][4] + g.log_cdf_d6(y)[3] + g.log_ccdf_d6(y)[1] + g.log_cdf(y) +
                     g.log_ccdf(y) + g.quantile(0.3 + 0.4 * i / n);
    CHECK(hr(i) == doctest::Approx(r).epsilon(1e-10));
  }
}

TEST_CASE("distrib beta: every order inside a kernel") { beta_kernel_case(); }

// ---------------------------------------------------------------- the references

TEST_CASE("distrib beta: the log-density and its derivatives against mpmath, phi from 1e-2 to 1e8") {
  // tests/data/reference/beta_logpdf.csv (tools/reference/distrib_beta.py): SymPy's derivatives of the definition
  // evaluated by mpmath at 200 and 300 bits, kept where the two agree. Columns y, mu, phi, K, b, hi, lo, b the
  // number of derivatives in phi.
  //
  // Bounds (provisional, see the head of this file), by what limits each order:
  //   K = 0, 1: the error is that of the terms the value is a sum of (lgamma at phi, alpha and beta, the
  //     digamma functions, the logarithms), eps times the sum of their magnitudes, 8 eps at most;
  //   K >= 2: the error in ulp of the largest component of the same (mu, phi, y, K); 40 ulp;
  //   the pure phi components of order 2 and more, relative to themselves: the leading monomials of the
  //     polygamma functions cancel in them algebraically, through detail_beta::pure_phi_combination, so
  //     the error carries no factor of phi; measured 13.4 ulp at most, bound 24 ulp.
  const auto rows = read_rows("beta_logpdf.csv", 7);
  CHECK(rows.size() > 10000);
  using key = std::tuple<double, double, double, int>;
  std::map<key, double> scale;
  for (const auto& s : rows) {
    auto& v = scale[key(num(s[1]), num(s[2]), num(s[0]), integer(s[3]))];
    v = std::fmax(v, std::fabs(num(s[5])));
  }
  double worst_scaled[7] = {}, worst_pure_phi[7] = {}, worst_terms = 0.0;
  for (const auto& s : rows) {
    const double y = num(s[0]), mu = num(s[1]), phi = num(s[2]);
    const int K = integer(s[3]), b = integer(s[4]);
    const double hi = num(s[5]), lo = num(s[6]);
    const bd d{mu, phi};
    const double nu = 1.0 - mu, a = mu * phi, be = nu * phi;
    INFO("y " << y << " mu " << mu << " phi " << phi << " K " << K << " b " << b);
    if (K == 0) {
      const double err = std::fabs((d.logpdf(y) - hi) - lo);
      const double terms = std::fabs(laplace::math::lgamma(phi)) + std::fabs(laplace::math::lgamma(a)) +
                           std::fabs(laplace::math::lgamma(be)) + std::fabs(a - 1.0) * std::fabs(std::log(y)) +
                           std::fabs(be - 1.0) * std::fabs(std::log1p(-y)) + 1.0;
      worst_terms = std::fmax(worst_terms, err / (eps * terms));
      CHECK(err <= 8.0 * eps * terms);
    } else if (K == 1) {
      const auto t = d.logpdf_d1(y);
      const double psa = laplace::math::digamma(a), psb = laplace::math::digamma(be);
      const double ly = std::log(y), l1y = std::log1p(-y);
      const double terms = b == 0 ? phi * (std::fabs(ly - l1y) + std::fabs(psa) + std::fabs(psb))
                                  : std::fabs(laplace::math::digamma(phi)) + mu * (std::fabs(ly) + std::fabs(psa)) +
                                        nu * (std::fabs(l1y) + std::fabs(psb));
      const double err = std::fabs((t[b] - hi) - lo);
      worst_terms = std::fmax(worst_terms, err / (eps * terms));
      CHECK(err <= 8.0 * eps * terms);
    } else {
      const auto t = lderiv(d, y, K);
      const double sc = scale[key(mu, phi, y, K)];
      const double err = std::fabs((t[b] - hi) - lo);
      const double u = err / ulp_at(sc);
      worst_scaled[K] = std::fmax(worst_scaled[K], u);
      CHECK(u <= 40.0);
      if (b == K && hi != 0.0) {
        const double own = err / ulp_at(hi);
        worst_pure_phi[K] = std::fmax(worst_pure_phi[K], own);
        CHECK(own <= 24.0);
      }
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    std::printf("beta logpdf: orders 0 and 1 worst error %.3g eps x terms\n", worst_terms);
    for (int K = 2; K <= 6; ++K)
      std::printf("beta logpdf K %d: scaled %.3g ulp of the order's scale, pure phi %.3g ulp of itself\n", K,
                  worst_scaled[K], worst_pure_phi[K]);
  }
}

TEST_CASE("distrib beta: the expected information and its derivatives against mpmath") {
  // tests/data/reference/beta_info.csv: columns kind, K, pair, b, mu, phi, hi, lo. kind 0: the quadrature of the
  // squared score (K = 0); kind 1: SymPy's derivatives of the closed form (K = 0..4), b the number of
  // derivatives in phi, pair 0 = (mu, mu), 1 = (mu, phi), 2 = (phi, phi).
  // Bounds (provisional): the error in units of eps times the largest component of the same (kind, K, mu,
  // phi), 64; the pure phi components (pair 2, b = K) relative to themselves, where the three terms cancel
  // algebraically through detail_beta::pure_phi_combination and so carry no factor of phi, 16 eps (measured 7.68).
  const auto rows = read_rows("beta_info.csv", 8);
  CHECK(rows.size() > 5000);
  using key = std::tuple<int, int, double, double>;
  std::map<key, double> scale;
  for (const auto& s : rows) {
    auto& v = scale[key(integer(s[0]), integer(s[1]), num(s[4]), num(s[5]))];
    v = std::fmax(v, std::fabs(num(s[6])));
  }
  double worst_scaled = 0.0, worst_pure = 0.0;
  int wk = -1, wkind = -1;
  double wmu = 0.0, wphi = 0.0;
  for (const auto& s : rows) {
    const int kind = integer(s[0]), K = integer(s[1]), pair = integer(s[2]), b = integer(s[3]);
    const double mu = num(s[4]), phi = num(s[5]), hi = num(s[6]), lo = num(s[7]);
    const bd d{mu, phi};
    const auto t = ideriv(d, K);
    const double got = t[pair * (K + 1) + b];
    const double err = std::fabs((got - hi) - lo);
    const double sc = scale[key(kind, K, mu, phi)];
    INFO("kind " << kind << " K " << K << " pair " << pair << " b " << b << " mu " << mu << " phi " << phi);
    worst_scaled = std::fmax(worst_scaled, err / (eps * sc));
    CHECK(err <= 64.0 * eps * sc);
    if (pair == 2 && b == K && hi != 0.0) {
      const double rp = err / (eps * std::fabs(hi));
      if (rp > worst_pure) {
        worst_pure = rp;
        wk = K;
        wkind = kind;
        wmu = mu;
        wphi = phi;
      }
      CHECK(err <= 16.0 * eps * std::fabs(hi));
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("beta info: worst %.3g eps of the largest component; pure phi %.3g eps of itself"
                " (kind %d K %d mu %g phi %g)\n",
                worst_scaled, worst_pure, wkind, wk, wmu, wphi);
}

TEST_CASE("distrib beta: the cdf family against mpmath") {
  // tests/data/reference/beta_cdf.csv: columns kind, y, mu, phi, hi, lo; kind 0 F, 1 1 - F, 2 log F, 3 log(1 - F).
  // Bounds (provisional): 5e-12 relative for F and 1 - F, 5e-12 max(1, |log|) absolute for the logarithms.
  const auto rows = read_rows("beta_cdf.csv", 6);
  CHECK(rows.size() > 3000);
  double worst[4] = {};
  for (const auto& s : rows) {
    const int kind = integer(s[0]);
    const double y = num(s[1]), mu = num(s[2]), phi = num(s[3]), hi = num(s[4]), lo = num(s[5]);
    const bd d{mu, phi};
    const double v = kind == 0 ? d.cdf(y) : kind == 1 ? d.ccdf(y) : kind == 2 ? d.log_cdf(y) : d.log_ccdf(y);
    const double err = std::fabs((v - hi) - lo);
    INFO("kind " << kind << " y " << y << " mu " << mu << " phi " << phi);
    if (kind < 2) {
      worst[kind] = std::fmax(worst[kind], err / std::fabs(hi));
      CHECK(err <= 5e-12 * std::fabs(hi));
    } else {
      const double sc = std::fmax(1.0, std::fabs(hi));
      worst[kind] = std::fmax(worst[kind], err / sc);
      CHECK(err <= 5e-12 * sc);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("beta cdf: worst F %.3g, 1 - F %.3g (relative), log F %.3g, log(1 - F) %.3g (absolute / max(1, |log|))\n",
                worst[0], worst[1], worst[2], worst[3]);
}

TEST_CASE("distrib beta: the derivatives of log F and log(1 - F) against mpmath, orders 1 to 6") {
  // tests/data/reference/beta_logcdf.csv: columns upper, y, mu, phi, K, b, hi, lo, from mpmath.diff of log F and
  // log(1 - F) in (mu, phi). Bound (provisional): 2e-11 of the largest component of the same (upper, y, mu, phi, K).
  const auto rows = read_rows("beta_logcdf.csv", 8);
  CHECK(rows.size() > 1000);
  using key = std::tuple<int, double, double, double, int>;
  std::map<key, double> scale;
  for (const auto& s : rows) {
    auto& v = scale[key(integer(s[0]), num(s[1]), num(s[2]), num(s[3]), integer(s[4]))];
    v = std::fmax(v, std::fabs(num(s[6])));
  }
  double worst[2][7] = {};
  for (const auto& s : rows) {
    const int upper = integer(s[0]), K = integer(s[4]), b = integer(s[5]);
    const double y = num(s[1]), mu = num(s[2]), phi = num(s[3]), hi = num(s[6]), lo = num(s[7]);
    const bd d{mu, phi};
    const auto t = cderiv(d, y, K, upper == 1);
    const double sc = scale[key(upper, y, mu, phi, K)];
    const double err = std::fabs((t[b] - hi) - lo);
    // Measured on gcc at 2ff5c21, as a fraction of the largest component of
    // the same (upper, y, mu, phi, K), worst over the grid by order:
    //   log F        4.3e-14  4.7e-14  1.2e-13  1.1e-12  2.4e-11  2.8e-10
    //   log(1 - F)   2.9e-15  1.8e-14  8.3e-14  1.3e-12  2.8e-11  1.8e-10
    // The bound is about one and a half times the worse tail of each order.
    // The two tails now agree in size, which they did not before 2ff5c21: the
    // upper one ran from 2.5e-07 to 3.2e-06 while the incomplete beta's
    // derivatives complemented a frame whose other shape was exactly one, and
    // the repair there moved it by five decades.
    // What still sets orders five and six is mu = 0.5, phi = 0.05, that is
    // alpha = beta = 0.025: the family's small-shape corner, where the pole
    // terms cancel in the mixed components. That is a documented limit and not
    // a defect, and the repair for it is the polygamma remainder layer, which
    // no family calls yet (step 4a of section 3.3 of piano_laplace.md).
    // CUDA is still to measure and the bound is to be the worse of the two.
    static const double cdf_bound[7] = {0.0, 7e-14, 7e-14, 2e-13, 2e-12, 4.5e-11, 4.5e-10};
    INFO("upper " << upper << " y " << y << " mu " << mu << " phi " << phi << " K " << K << " b " << b);
    worst[upper][K] = std::fmax(worst[upper][K], err / sc);
    CHECK(err <= cdf_bound[K] * sc);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("beta log%s K %d: worst %.3g of the largest component\n", u ? "(1 - F)" : "F", K, worst[u][K]);
}
