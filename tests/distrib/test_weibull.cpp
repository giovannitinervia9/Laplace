// weibull_distrib (include/laplace/distrib/weibull.hpp): the value and every derivative order of
// log f, log F and log(1 - F) against mpmath (tools/reference/distrib_weibull.py, built from the
// definition of the family), the mixed (y, theta) derivatives against mpmath, the exact twin at
// sigma = 1 (the exponential at scale m), the derivatives against differences of the order below,
// the survival derivatives where 1 - F underflows, the quantile by round trips from p and from log p
// in both tails, the closed expected information against the generic quadrature and against mpmath,
// its orthogonality at sigma = 1, and a kernel launch of the highest orders.
//
// Accuracy measure of the reference tests ("scaled ulp"). A derivative of order K of l with a indices
// in m, b in sigma and j in y is multiplied by m^a sigma^b y^j, so that the components of one
// order have one unit; its error is divided by the ulp of the largest such component at the same
// point, and by the conditioning of the point: h = sigma log(y/b) carries the absolute error
// eps (1 + sigma |log(y/m)|) from the rounding of the logarithm, and e^h magnifies it, so a quantity
// proportional to T = e^h (log(1 - F), the density in the right tail) is divided by that factor and
// a quantity proportional to e^-T (log F for T above 1) by that factor times 1 + T; l_m = sigma
// expm1(h)/m has its root at y = b, where its conditioning is sigma times larger again.
// LAPLACE_PRINT_ERRORS=1 prints the worst values per order.
#include <doctest/doctest.h>

#include <laplace/distrib/exponential.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/weibull.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included
using laplace_test::hide;

namespace {

constexpr double kEulerGamma = 0.57721566490153286061;
constexpr double kPi2Over6 = 1.6449340668482264365;

// the scale b = m/Gamma(1 + 1/sigma) through the C library, for moderate parameters only
double scale_b(double m, double sigma) { return m / std::tgamma(1.0 + 1.0 / sigma); }

// the density of the textbook form (k/b)(y/b)^(k-1) exp(-(y/b)^k)
double textbook_density(double m, double k, double y) {
  const double b = scale_b(m, k);
  return (k / b) * std::pow(y / b, k - 1.0) * std::exp(-std::pow(y / b, k));
}

double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

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

struct wrow {
  double m = 0, s = 0, y = 0, hi = 0, lo = 0;
  int J = 0, K = 0, b = 0;
};

// weibull_logpdf.csv, weibull_log_cdf.csv, weibull_log_ccdf.csv: m,sigma,y,K,b,hi,lo;
// weibull_dy.csv: m,sigma,y,J,K,b,hi,lo
std::vector<wrow> read_rows(const char* name, bool with_j) {
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::vector<wrow> rows;
  if (!std::fgets(line, sizeof line, f)) return rows;  // header
  while (std::fgets(line, sizeof line, f)) {
    const auto c = split_csv(line);
    wrow r;
    r.m = std::strtod(c[0].c_str(), nullptr);
    r.s = std::strtod(c[1].c_str(), nullptr);
    r.y = std::strtod(c[2].c_str(), nullptr);
    int i = 3;
    if (with_j) r.J = std::atoi(c[i++].c_str());
    r.K = std::atoi(c[i++].c_str());
    r.b = std::atoi(c[i++].c_str());
    r.hi = std::strtod(c[i++].c_str(), nullptr);
    r.lo = std::strtod(c[i].c_str(), nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

double log_ratio(double y, double m) {
  const double q = y / m;
  return (q > 1e-250 && q < 1e250) ? std::log(q) : std::log(y) - std::log(m);
}

// file: 0 log f, 1 log F, 2 log(1 - F), 3 (y, theta) derivatives of log f
double library_value(const ld::weibull_distrib& d, int file, const wrow& r) {
  if (file == 3) return yderiv(d, r.y, r.J, r.K)[r.b];
  if (r.K == 0) return file == 0 ? d.logpdf(r.y) : (file == 1 ? d.log_cdf(r.y) : d.log_ccdf(r.y));
  if (file == 0) return lderiv(d, r.y, r.K)[r.b];
  return cderiv(d, r.y, r.K, file == 2)[r.b];
}

// the worst scaled ulp per order K (J included for the mixed file) of one reference file, and a check of the bounds
void check_reference(const char* name, int file, const double* bound) {
  const auto rows = read_rows(name, file == 3);
  CHECK(rows.size() > 200);
  // the largest weighted |value| of each (point, J, K) group
  std::map<std::tuple<double, double, double, int, int>, double> scale;
  const auto weight = [](const wrow& r) { return std::pow(r.m, r.K - r.b) * std::pow(r.s, r.b) * std::pow(r.y, r.J); };
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.m, r.s, r.y, r.J, r.K)];
    s = std::fmax(s, std::fabs(r.hi) * weight(r));
  }
  double worst[7] = {};
  double worst_y[7] = {}, worst_s[7] = {};
  for (const auto& r : rows) {
    const ld::weibull_distrib d{r.m, r.s};
    const double v = library_value(d, file, r);
    const double sc = scale[std::make_tuple(r.m, r.s, r.y, r.J, r.K)];
    // a group of mixed derivatives that is zero to rounding (a root of the polynomial at the point) has no scale
    if (file == 3 && sc < 1e-10) continue;
    const double err = std::fabs((v - r.hi) - r.lo) * weight(r);
    double cond = 1.0 + r.s * std::fabs(log_ratio(r.y, r.m));
    if (file == 1) cond *= 1.0 + std::exp(d.hazard_log(r.y));
    if (file != 3 && r.K == 1 && r.b == 0) cond *= std::fmax(1.0, r.s);
    const double e = err / ulp_of(sc) / cond;
    const int order = file == 3 ? r.J + r.K : r.K;
    if (e > worst[order]) worst[order] = e, worst_y[order] = r.y, worst_s[order] = r.s;
    INFO(name << ": J " << r.J << " K " << r.K << " b " << r.b << " m " << r.m << " sigma " << r.s << " y " << r.y
              << " scaled ulp " << e);
    CHECK(e <= bound[order]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int k = 0; k <= 6; ++k)
      std::printf("weibull %s order %d: worst %.3g scaled ulp (sigma %.6g, y %.6g)\n", name, k, worst[k], worst_s[k],
                  worst_y[k]);
}

}  // namespace

TEST_CASE("weibull: descriptors and kinks") {
  CHECK(ld::weibull_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::weibull_distrib::n_params == 2);
  CHECK(ld::weibull_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::weibull_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::weibull_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::weibull_distrib::params[j].moves_support == false);
    CHECK(ld::weibull_distrib::params[j].lower == 0.0);
    CHECK(ld::weibull_distrib::params[j].upper == ld::distrib_infinity);
  }
  const ld::weibull_distrib d{2.5, 1.7};
  CHECK(d.param_kinks(0, 1.0).count == 0);
  CHECK(d.param_kinks(1, 1.0).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.center() == 2.5);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == ld::distrib_infinity);
}

TEST_CASE("weibull: density against the textbook form, mean and standard deviation") {
  for (double m : {0.4, 3.0, 250.0})
    for (double k : {0.6, 1.0, 1.7, 4.0})
      for (double u : {0.05, 0.4, 1.0, 2.2}) {
        const double y = m * u;
        const ld::weibull_distrib d{m, k};
        INFO("m " << m << " k " << k << " y " << y);
        CHECK(d.logpdf(y) == doctest::Approx(std::log(textbook_density(m, k, y))).epsilon(1e-11).scale(1.0));
      }
  // the scale() is the standard deviation m sqrt(Gamma(1 + 2/k)/Gamma(1 + 1/k)^2 - 1)
  for (double k : {0.7, 1.0, 2.0, 5.0}) {
    const ld::weibull_distrib d{3.0, k};
    const double sd = 3.0 * std::sqrt(std::tgamma(1.0 + 2.0 / k) / (std::tgamma(1.0 + 1.0 / k) * std::tgamma(1.0 + 1.0 / k)) - 1.0);
    CHECK(d.scale() == doctest::Approx(sd).epsilon(1e-12));
  }
  // large sigma: the standard deviation is about m pi/(sigma sqrt 6) (no cancellation of the Gamma factors)
  const ld::weibull_distrib big{2.0, 1e4};
  CHECK(big.scale() == doctest::Approx(2.0 * 3.14159265358979323846 / (1e4 * std::sqrt(6.0))).epsilon(2e-4));
  // the mean is m: integral of y f(y) by the exponential-variable form, E[Y] = b Gamma(1 + 1/k), b from log b
  for (double k : {0.5, 2.0, 9.0}) {
    const ld::weibull_distrib d{7.0, k};
    CHECK(std::exp(d.log_scale() + std::lgamma(1.0 + 1.0 / k)) == doctest::Approx(7.0).epsilon(1e-12));
  }
}

TEST_CASE("weibull: first derivatives in closed form") {
  // l_m = sigma (T - 1)/m, l_mm = sigma (-sigma T - (T - 1))/m^2, T = (y/b)^sigma
  for (double sigma : {0.8, 1.0, 2.5})
    for (double y : {0.3, 1.0, 2.7}) {
      const double m = 1.6;
      const ld::weibull_distrib d{m, sigma};
      const double T = std::pow(y / scale_b(m, sigma), sigma);
      CHECK(d.logpdf_d1(y)[0] == doctest::Approx(sigma * (T - 1.0) / m).epsilon(1e-12).scale(1.0));
      CHECK(d.logpdf_d2(y)[0] == doctest::Approx(sigma * (-sigma * T - (T - 1.0)) / (m * m)).epsilon(1e-12).scale(1.0));
      // l_sigma = (1 - T)(C + LG) + (T - 1) psi(1 + 1/sigma)/sigma + 1/sigma, the second form of the plan
      const double LG = std::lgamma(1.0 + 1.0 / sigma);
      const double psi = ld::detail_weibull::digamma_shift(sigma) + (1.0 - kEulerGamma);
      const double ls = (1.0 - T) * (std::log(y / m) + LG) + (T - 1.0) * psi / sigma + 1.0 / sigma;
      CHECK(d.logpdf_d1(y)[1] == doctest::Approx(ls).epsilon(1e-11).scale(1.0));
    }
}

TEST_CASE("weibull: derivatives against differences") {
  for (double y : {0.2, 1.1, 3.0}) check_derivs<ld::weibull_distrib>({1.5, 1.3}, y, 1e-6);
  for (double y : {5.0, 40.0, 130.0}) check_derivs<ld::weibull_distrib>({40.0, 0.7}, y, 1e-6);
  for (double y : {1.9, 2.4}) check_derivs<ld::weibull_distrib>({2.0, 6.0}, y, 1e-6);
  for (double y : {0.9, 1.05}) check_derivs<ld::weibull_distrib>({1.0, 1.0}, y, 1e-6);
}

TEST_CASE("weibull: mixed (y, theta) derivatives against differences") {
  for (double y : {0.4, 1.3, 3.0}) check_y_derivs<ld::weibull_distrib>({1.5, 1.3}, y, 1e-3 * y, 1e-6);
  for (double y : {15.0, 60.0}) check_y_derivs<ld::weibull_distrib>({40.0, 0.7}, y, 1e-3 * y, 1e-6);
}

TEST_CASE("weibull: derivatives in y and the support boundary") {
  // y = 0: the derivative in y is infinite for sigma != 1 and -1/m (the exponential at scale m) at sigma = 1
  const ld::weibull_distrib half{2.0, 0.5}, one{2.0, 1.0}, three{2.0, 3.0};
  CHECK(half.logpdf_dy_dk<1, 0>(0.0)[0] == ld::distrib_infinity);
  CHECK(three.logpdf_dy_dk<1, 0>(0.0)[0] == ld::distrib_infinity);
  CHECK(one.logpdf_dy_dk<1, 0>(0.0)[0] == -0.5);
  CHECK(one.logpdf_dy_dk<2, 0>(0.0)[0] == 0.0);
  CHECK(half.logpdf(0.0) == -ld::distrib_infinity);
  CHECK(std::isnan(half.logpdf_d1(0.0)[0]));
}

TEST_CASE("weibull: exact twin of the exponential at sigma = 1") {
  // at sigma = 1 the family is the exponential at scale b = m (Gamma(2) = 1): every pure-m quantity
  double worst = 0.0;
  // `scale` is the natural size of the quantity, and the criterion is relative
  // to it with an absolute floor of 8 eps scale on top. Both parts earned
  // their place on 2026-11-15.
  //
  // The relative part was written as |w - e| / (|e| + scale) with scale passed
  // as 0, which divides by zero whenever the reference value is exactly zero
  // and returned inf; that is a defect of the comparison, not of either
  // family. The floor is for the quantities whose exact value *is* zero by an
  // exact cancellation, where no relative criterion can hold: at sigma = 1 the
  // weibull writes log f as -log y + h with h = log(y/b), so its mixed
  // y-derivatives cancel term by term and what survives is the rounding of
  // terms of size (J-1)!/(y^J m^K), while the exponential, whose log f is
  // -log m - y/m, returns an exact 0. The twin then compares noise against
  // zero, and the honest bound is the size of what cancelled.
  //
  // The same case found a real defect in exponential_distrib::log_cdf, which
  // returned exactly 0 for u above about 36 where the weibull gave the correct
  // -e^-u; that is repaired in exponential.hpp.
  const auto cmp = [&](const char* what, double w, double e, double scale) {
    const double den = std::fmax(std::fabs(e), scale);
    const double tol = 2e-12 * den + 8.0 * std::numeric_limits<double>::epsilon() * scale;
    const double err = den > 0.0 ? std::fabs(w - e) / den : std::fabs(w - e);
    worst = std::fmax(worst, err);
    INFO("quantity " << std::string(what) << ": weibull " << w << " against exponential " << e
                     << ", scale " << scale << ", tolerance " << tol);
    CHECK(std::fabs(w - e) <= tol);
  };
  for (double m : {0.5, 3.0, 1e3})
    for (double u : {0.01, 0.5, 1.3, 7.0, 40.0}) {
      const double y = m * u;
      const ld::weibull_distrib w{m, 1.0};
      const ld::exponential_distrib e{m};
      INFO("m " << m << " y " << y);
      cmp("logpdf(y)", w.logpdf(y), e.logpdf(y), 1.0);
      cmp("cdf(y)", w.cdf(y), e.cdf(y), 0.0);
      cmp("ccdf(y)", w.ccdf(y), e.ccdf(y), 0.0);
      cmp("log_cdf(y)", w.log_cdf(y), e.log_cdf(y), 0.0);
      cmp("log_ccdf(y)", w.log_ccdf(y), e.log_ccdf(y), 0.0);
      cmp("logpdf_d1(y)[0]", w.logpdf_d1(y)[0], e.logpdf_d1(y)[0], 1.0 / m);
      cmp("logpdf_d2(y)[0]", w.logpdf_d2(y)[0], e.logpdf_d2(y)[0], 1.0 / (m * m));
      cmp("logpdf_d3(y)[0]", w.logpdf_d3(y)[0], e.logpdf_d3(y)[0], 2.0 / (m * m * m));
      cmp("logpdf_d4(y)[0]", w.logpdf_d4(y)[0], e.logpdf_d4(y)[0], 6.0 / std::pow(m, 4));
      cmp("logpdf_d5(y)[0]", w.logpdf_d5(y)[0], e.logpdf_d5(y)[0], 24.0 / std::pow(m, 5));
      cmp("logpdf_d6(y)[0]", w.logpdf_d6(y)[0], e.logpdf_d6(y)[0], 120.0 / std::pow(m, 6));
      cmp("log_ccdf_d1(y)[0]", w.log_ccdf_d1(y)[0], e.log_ccdf_d1(y)[0], 0.0);
      cmp("log_ccdf_d2(y)[0]", w.log_ccdf_d2(y)[0], e.log_ccdf_d2(y)[0], 0.0);
      cmp("log_ccdf_d3(y)[0]", w.log_ccdf_d3(y)[0], e.log_ccdf_d3(y)[0], 0.0);
      cmp("log_ccdf_d4(y)[0]", w.log_ccdf_d4(y)[0], e.log_ccdf_d4(y)[0], 0.0);
      cmp("log_ccdf_d5(y)[0]", w.log_ccdf_d5(y)[0], e.log_ccdf_d5(y)[0], 0.0);
      cmp("log_ccdf_d6(y)[0]", w.log_ccdf_d6(y)[0], e.log_ccdf_d6(y)[0], 0.0);
      cmp("log_cdf_d1(y)[0]", w.log_cdf_d1(y)[0], e.log_cdf_d1(y)[0], 0.0);
      cmp("log_cdf_d2(y)[0]", w.log_cdf_d2(y)[0], e.log_cdf_d2(y)[0], 0.0);
      cmp("log_cdf_d3(y)[0]", w.log_cdf_d3(y)[0], e.log_cdf_d3(y)[0], 0.0);
      cmp("log_cdf_d4(y)[0]", w.log_cdf_d4(y)[0], e.log_cdf_d4(y)[0], 0.0);
      cmp("log_cdf_d5(y)[0]", w.log_cdf_d5(y)[0], e.log_cdf_d5(y)[0], 0.0);
      cmp("log_cdf_d6(y)[0]", w.log_cdf_d6(y)[0], e.log_cdf_d6(y)[0], 0.0);
      cmp("logpdf_dy_dk<1", w.logpdf_dy_dk<1, 0>(y)[0], e.logpdf_dy_dk<1, 0>(y)[0], 0.0);
      cmp("logpdf_dy_dk<1", w.logpdf_dy_dk<1, 3>(y)[0], e.logpdf_dy_dk<1, 3>(y)[0], 0.0);
      // These two are exact roots at sigma = 1, so the scale is the size of the
      // terms that cancel, (J - 1)! / (y^J m^K), and not a power of m alone.
      cmp("logpdf_dy_dk<2, 4>", w.logpdf_dy_dk<2, 4>(y)[0], e.logpdf_dy_dk<2, 4>(y)[0],
          1.0 / (std::pow(y, 2) * std::pow(m, 4)));
      cmp("logpdf_dy_dk<5, 1>", w.logpdf_dy_dk<5, 1>(y)[0], e.logpdf_dy_dk<5, 1>(y)[0],
          24.0 / (std::pow(y, 5) * m));
      cmp("expected_info()[0]", w.expected_info()[0], e.expected_info()[0], 0.0);
      cmp("expected_info_d1().pair[0][0]", w.expected_info_d1().pair[0][0], e.expected_info_d1().pair[0][0], 0.0);
      cmp("expected_info_d4().pair[0][0]", w.expected_info_d4().pair[0][0], e.expected_info_d4().pair[0][0], 0.0);
    }
  for (double p : {1e-9, 0.2, 0.5, 0.97, 1.0 - 0x1p-40}) {
    const ld::weibull_distrib w{1.7, 1.0};
    const ld::exponential_distrib e{1.7};
    cmp("quantile(p)", w.quantile(p), e.quantile(p), 0.0);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("weibull twin of the exponential: worst relative difference %.3g\n", worst);
}

TEST_CASE("weibull: value and derivatives of log f against mpmath") {
  // bounds provisional (about 4 times the emulated worst); measured values are printed by LAPLACE_PRINT_ERRORS
  const double bound[7] = {80.0, 25.0, 25.0, 20.0, 25.0, 40.0, 50.0};
  check_reference("weibull_logpdf.csv", 0, bound);
}

TEST_CASE("weibull: value and derivatives of log F against mpmath") {
  const double bound[7] = {20.0, 20.0, 30.0, 40.0, 150.0, 600.0, 900.0};
  check_reference("weibull_log_cdf.csv", 1, bound);
}

TEST_CASE("weibull: value and derivatives of log(1 - F) against mpmath") {
  const double bound[7] = {30.0, 25.0, 25.0, 20.0, 25.0, 40.0, 50.0};
  check_reference("weibull_log_ccdf.csv", 2, bound);
}

TEST_CASE("weibull: mixed (y, theta) derivatives against mpmath") {
  // Measured on the gcc build on 2026-11-15 with a margin of about half again.
  // Order five was 137.1 scaled ulp at m = 1, sigma = 0.01, y = 1.36e-28,
  // where the agent's emulation had predicted under 100; the others held.
  const double bound[7] = {0.0, 20.0, 30.0, 30.0, 300.0, 210.0, 250.0};
  check_reference("weibull_dy.csv", 3, bound);
}

TEST_CASE("weibull: log(1 - F) derivatives where 1 - F underflows") {
  // (m, sigma, y) = (100, 8, 480): T = (y/b)^sigma is about 1.7e5, so 1 - F = e^-T is 0 in a double while
  // log(1 - F) = -T is finite. The pure m derivatives have the closed form
  //   d^K(-T)/dm^K = -T (-1)^K sigma (sigma + 1) ... (sigma + K - 1)/m^K
  // because T is proportional to m^-sigma, so d^K m^-sigma / dm^K carries
  // (-sigma)(-sigma - 1) ... (-sigma - K + 1) = (-1)^K sigma (sigma + 1) ... .
  // The (-1)^K was missing here and from this comment, which is why only the
  // odd orders failed while the magnitudes agreed to six digits: at K = 1 the
  // truth is +sigma T/m = 8 * 174336 / 100 = 13946.9, which is what the
  // library returns.
  const ld::weibull_distrib d{100.0, 8.0};
  const double y = 480.0;
  CHECK(d.ccdf(y) == 0.0);
  const double lc = d.log_ccdf(y);
  CHECK(std::isfinite(lc));
  CHECK(lc < -1e5);
  const double T = -lc;
  const double c[6] = {d.log_ccdf_d1(y)[0], d.log_ccdf_d2(y)[0], d.log_ccdf_d3(y)[0],
                       d.log_ccdf_d4(y)[0], d.log_ccdf_d5(y)[0], d.log_ccdf_d6(y)[0]};
  double rise = 1.0;
  for (int K = 1; K <= 6; ++K) {
    rise *= (8.0 + K - 1.0) / 100.0;
    INFO("K " << K);
    CHECK(std::isfinite(c[K - 1]));
    const double sign = (K % 2) ? -1.0 : 1.0;  // (-1)^K
    CHECK(c[K - 1] == doctest::Approx(-T * sign * rise).epsilon(1e-12));
  }
  // the sigma derivatives are finite too and obey the differences of the order below
  for (int K = 1; K <= 6; ++K) {
    const auto v = cderiv(d, y, K, true);
    for (double e : v) CHECK(std::isfinite(e));
  }
  check_cdf_derivs<ld::weibull_distrib>({100.0, 4.0}, 600.0, true, 1e-6);  // T about 876: 1 - F underflows
  check_cdf_derivs<ld::weibull_distrib>({100.0, 4.0}, 600.0, false, 1e-6);
}

TEST_CASE("weibull: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {1e-3, 0.3, 1.0, 2.0, 5.0}) check_cdf_derivs<ld::weibull_distrib>({1.2, 1.7}, y, upper, 1e-6);
    for (double y : {0.05, 8.0, 400.0}) check_cdf_derivs<ld::weibull_distrib>({30.0, 0.6}, y, upper, 1e-6);
    for (double y : {28.0, 33.0, 36.0}) check_cdf_derivs<ld::weibull_distrib>({30.0, 7.0}, y, upper, 1e-6);
  }
}

TEST_CASE("weibull: cdf, ccdf, logarithms") {
  const ld::weibull_distrib d{1.4, 1.9};
  for (double y : {1e-6, 0.1, 0.8, 1.4, 3.0, 6.0}) {
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
    CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
    const double b = scale_b(1.4, 1.9);
    CHECK(d.cdf(y) == doctest::Approx(-std::expm1(-std::pow(y / b, 1.9))).epsilon(1e-12));
  }
  // y <= 0
  CHECK(d.cdf(0.0) == 0.0);
  CHECK(d.ccdf(0.0) == 1.0);
  CHECK(d.log_cdf(0.0) == -ld::distrib_infinity);
  CHECK(d.log_ccdf(0.0) == 0.0);
  // T underflows (y tiny at large sigma): log F = h = log T stays finite, and log(1 - F) = -T = -0
  const ld::weibull_distrib sharp{1.0, 100.0};
  const double y = 1e-4;  // T = (y/b)^100, about 1e-400
  const double lF = sharp.log_cdf(y);
  CHECK(std::isfinite(lF));
  CHECK(lF == doctest::Approx(100.0 * (std::log(y) + std::lgamma(1.01))).epsilon(1e-12));
  CHECK(sharp.log_ccdf(y) == 0.0);
  // F = 1 - e^-T with T large: log(1 - F) = -T finite far past the underflow of 1 - F
  CHECK(std::isfinite(d.log_ccdf(1e3)));
  CHECK(d.log_ccdf(1e3) < -1e5);
}

TEST_CASE("weibull: quantile in both tails and from log p") {
  for (double m : {0.02, 1.0, 400.0})
    for (double sigma : {0.45, 1.0, 2.2, 15.0}) {
      const ld::weibull_distrib d{m, sigma};
      INFO("m " << m << " sigma " << sigma);
      // round trip p -> y -> p, relative error of the probability
      for (double p : {1e-12, 1e-4, 0.1, 0.5, 0.9, 0.999}) {
        const double y = d.quantile(p);
        CHECK(d.cdf(y) == doctest::Approx(p).epsilon(1e-11));
      }
      // log p deep in the lower tail: log F(Q) = log p
      for (double lp : {-1e-3, -1.0, -40.0, -300.0}) {
        const double y = d.quantile_from_log(lp);
        if (y == 0.0 || !std::isfinite(y)) continue;
        CHECK(d.log_cdf(y) == doctest::Approx(lp).epsilon(1e-10));
      }
      // log q = log(1 - p) deep in the upper tail: log(1 - F(Q)) = log q
      for (double lq : {-1e-3, -1.0, -40.0, -300.0, -5000.0}) {
        const double y = d.quantile_from_log_upper(lq);
        if (y == 0.0 || !std::isfinite(y)) continue;
        CHECK(d.log_ccdf(y) == doctest::Approx(lq).epsilon(1e-10));
      }
      // the three entries agree
      CHECK(d.quantile_from_log(std::log(0.3)) == doctest::Approx(d.quantile(0.3)).epsilon(1e-12));
      CHECK(d.quantile_from_log_upper(std::log(0.3)) == doctest::Approx(d.quantile(0.7)).epsilon(1e-12));
    }
  // the median: b (log 2)^(1/k)
  const ld::weibull_distrib d{3.0, 2.0};
  CHECK(d.quantile(0.5) == doctest::Approx(scale_b(3.0, 2.0) * std::sqrt(std::log(2.0))).epsilon(1e-13));
  // the end points
  CHECK(d.quantile(0.0) == 0.0);
  CHECK(d.quantile(1.0) == ld::distrib_infinity);
  CHECK(std::isnan(d.quantile(-0.1)));
  CHECK(d.quantile_from_log(0.0) == ld::distrib_infinity);
  CHECK(d.quantile_from_log_upper(0.0) == 0.0);
  // the generic root-finding route on the cdf gives the same quantile as the closed form
  for (double p : {1e-6, 0.3, 0.9, 1.0 - 1e-6}) {
    const ld::weibull_distrib w{2.0, 1.4};
    CHECK(ld::distrib_quantile(hide(w), p) == doctest::Approx(w.quantile(p)).epsilon(1e-9));
  }
  // small sigma: log b = log m - lgamma(1 + 1/sigma) is finite where b itself underflows (sigma = 1e-3: -5912)
  const ld::weibull_distrib tiny{1.0, 1e-3};
  CHECK(std::isfinite(tiny.log_scale()));
  CHECK(tiny.log_scale() == doctest::Approx(-std::lgamma(1001.0)).epsilon(1e-13));
}

namespace {
// the closed information against the generic score-form quadrature over y, orders K = 0, 1, 2
template <int K, class D>
double info_by_integral_error(const D& d) {
  const auto closed = [&](const auto& t) {
    std::vector<double> r;
    if constexpr (K == 0) {
      for (int i = 0; i < 3; ++i) r.push_back(t[i]);
    } else {
      for (int p = 0; p < 3; ++p)
        for (int i = 0; i < ld::sym_tensor_size(2, K); ++i) r.push_back(t.pair[p][i]);
    }
    return r;
  };
  std::vector<double> ex;
  if constexpr (K == 0) ex = closed(d.expected_info());
  else ex = closed(d.template expected_info_dk<K>());
  const auto gen = closed(ld::expected_info_by_integral<K, ld::info_form::score>(hide(d)));
  double scale = 0.0;
  for (double e : ex) scale = std::fmax(scale, std::fabs(e));
  double worst = 0.0;
  for (size_t i = 0; i < ex.size(); ++i) {
    if (std::isnan(gen[i])) return INFINITY;
    worst = std::fmax(worst, std::fabs(gen[i] - ex[i]) / scale);
  }
  return worst;
}
}  // namespace

TEST_CASE("weibull: expected information against the generic quadrature") {
  for (double sigma : {0.8, 1.3, 3.0, 8.0}) {
    const ld::weibull_distrib d{1.5, sigma};
    INFO("sigma " << sigma);
    CHECK(info_by_integral_error<0>(d) <= 1e-9);
    CHECK(info_by_integral_error<1>(d) <= 1e-8);
    CHECK(info_by_integral_error<2>(d) <= 1e-7);
  }
  // the closed form itself: I_mm = sigma^2/m^2, I_sigma,sigma = (D^2 + pi^2/6)/sigma^2, D = psi(1 + 1/sigma) - (1 - gamma)
  for (double sigma : {0.5, 2.0, 7.0}) {
    const ld::weibull_distrib d{2.5, sigma};
    const auto e = d.expected_info();
    // psi(1 + 1/sigma) from the C library's lgamma by a central difference of high order
    const double x = 1.0 / sigma, h = 1e-3;
    const auto lg = [](double a) { return std::lgamma(a); };
    const double psi = (-lg(1.0 + x + 2 * h) + 8 * lg(1.0 + x + h) - 8 * lg(1.0 + x - h) + lg(1.0 + x - 2 * h)) / (12 * h);
    const double D = psi - (1.0 - kEulerGamma);
    CHECK(e[0] == doctest::Approx(sigma * sigma / 6.25).epsilon(1e-15));
    CHECK(e[1] == doctest::Approx(D / 2.5).epsilon(1e-9).scale(1.0));
    CHECK(e[2] == doctest::Approx((D * D + kPi2Over6) / (sigma * sigma)).epsilon(1e-9));
  }
  check_info_derivs<ld::weibull_distrib>({1.5, 1.3}, 1e-6);
  check_info_derivs<ld::weibull_distrib>({0.4, 0.7}, 1e-6);
  check_info_derivs<ld::weibull_distrib>({3.0, 6.0}, 1e-6);
}

TEST_CASE("weibull: the parameters are orthogonal at sigma = 1") {
  // D = psi(1 + 1/sigma) - (1 - gamma) is exactly 0 at sigma = 1 (the series in d = 1/sigma - 1) and
  // has the series psi'(2) d - ... elsewhere nearby
  for (double m : {0.01, 1.0, 300.0}) {
    CHECK(ld::weibull_distrib{m, 1.0}.expected_info()[1] == 0.0);
    CHECK(ld::weibull_distrib{m, 1.0}.expected_info_d1().pair[1][0] == 0.0);  // d/dm of D/m
    for (double dsig : {1e-9, 1e-5, 1e-2, -1e-9, -1e-5, -1e-2}) {
      const double sigma = 1.0 + dsig;
      const double d = (1.0 - sigma) / sigma;
      // psi(2 + d) - psi(2) = (zeta(2) - 1) d - (zeta(3) - 1) d^2 + (zeta(4) - 1) d^3 - ...
      const double series = (kPi2Over6 - 1.0) * d - 0.2020569031595942854 * d * d + 0.08232323371113819152 * d * d * d -
                            0.03692775514336992633 * d * d * d * d + 0.01734306198444913971 * std::pow(d, 5);
      const double got = ld::weibull_distrib{m, sigma}.expected_info()[1] * m;
      INFO("m " << m << " sigma " << sigma);
      CHECK(got == doctest::Approx(series).epsilon(1e-9 + 0.0083 * std::pow(std::fabs(d), 5)));
    }
  }
}

TEST_CASE("weibull: expected information against mpmath") {
  const auto rows = [] {
    const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/weibull_info.csv";
    std::FILE* f = std::fopen(path.c_str(), "r");
    REQUIRE(f != nullptr);
    char line[512];
    std::vector<std::vector<double>> r;
    if (!std::fgets(line, sizeof line, f)) return r;
    while (std::fgets(line, sizeof line, f)) {
      const auto c = split_csv(line);
      std::vector<double> v;
      for (const auto& s : c) v.push_back(std::strtod(s.c_str(), nullptr));
      r.push_back(v);
    }
    std::fclose(f);
    return r;
  }();
  CHECK(rows.size() >= 20);
  double worst[3] = {};
  for (const auto& r : rows) {
    const ld::weibull_distrib d{r[0], r[1]};
    const auto e = d.expected_info();
    // the natural scales: I_mm, sqrt(I_mm I_ss) and I_ss
    const double ref[3] = {r[2] + r[3], r[4] + r[5], r[6] + r[7]};
    const double sc[3] = {ref[0], std::sqrt(ref[0] * ref[2]), ref[2]};
    for (int i = 0; i < 3; ++i) {
      const double err = std::fabs(e[i] - ref[i]) / sc[i];
      worst[i] = std::fmax(worst[i], err);
      INFO("m " << r[0] << " sigma " << r[1] << " component " << i);
      CHECK(err <= 3e-14);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("weibull expected information vs mpmath: worst relative to the natural scale %.3g, %.3g, %.3g\n", worst[0],
                worst[1], worst[2]);
}

TEST_CASE("weibull: start_value") {
  // a sample from the exponential-like regime: mean 2, sd 2 gives shape (1)^-1.086 = 1
  ld::distrib_sample_summary s;
  s.mean = 2.0;
  s.variance = 4.0;
  const ld::weibull_distrib w = ld::distrib_start_value(ld::weibull_distrib{1.0, 1.0}, s);
  CHECK(w.m == 2.0);
  CHECK(w.sigma == doctest::Approx(1.0).epsilon(1e-14));
  // sd/mean = 0.5: shape = 0.5^-1.086
  s.variance = 1.0;
  CHECK(ld::distrib_start_value(ld::weibull_distrib{1.0, 1.0}, s).sigma == doctest::Approx(std::pow(0.5, -1.086)).epsilon(1e-14));
  // very large dispersion: shape floored at 1e-2
  s.variance = 1e300;
  CHECK(ld::distrib_start_value(ld::weibull_distrib{1.0, 1.0}, s).sigma == 1e-2);
  // degenerate samples stay inside the domain
  s.variance = 0.0;
  const auto z = ld::distrib_start_value(ld::weibull_distrib{1.0, 1.0}, s);
  CHECK(z.m == 2.0);
  CHECK(z.sigma == 1.0);
  s.mean = -1.0;
  s.variance = 1.0;
  const auto n = ld::distrib_start_value(ld::weibull_distrib{1.0, 1.0}, s);
  CHECK(n.m == 1.0);
  CHECK(n.sigma == 1.0);
  const auto from_data = ld::summarize_sample({1.2, 0.8, 2.5, 3.1, 0.4, 1.9, 1.1, 2.2});
  const auto st = ld::distrib_start_value(ld::weibull_distrib{1.0, 1.0}, from_data);
  CHECK(st.m > 0.0);
  CHECK(st.sigma > 0.0);
  CHECK(std::isfinite(st.sigma));
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp (doctest gives each case an internal-linkage
// function whose number restarts per translation unit, and nvcc tags every extended lambda with that address).
void weibull_kernel_case() {
  const int n = 24;
  Kokkos::View<double* [40]> out("out", n);
  Kokkos::parallel_for(
      "distrib_weibull", n, KOKKOS_LAMBDA(const int i) {
        const double sig[4] = {0.3, 1.0, 2.5, 12.0};
        const double s = sig[i % 4];
        const ld::weibull_distrib w{1.7, s};
        const double y = 0.2 + 0.25 * i;
        const auto d6 = w.logpdf_d6(y);
        const auto c6 = w.log_cdf_d6(y);
        const auto u6 = w.log_ccdf_d6(y);
        const auto j = w.expected_info_d4();
        int c = 0;
        for (int b = 0; b <= 6; ++b) out(i, c++) = d6[b];
        for (int b = 0; b <= 6; ++b) out(i, c++) = c6[b];
        for (int b = 0; b <= 6; ++b) out(i, c++) = u6[b];
        for (int b = 0; b <= 4; ++b) out(i, c++) = j.pair[1][b] + j.pair[2][b];
        out(i, c++) = w.logpdf_dy_dk<2, 3>(y)[1];
        out(i, c++) = w.logpdf(y);
        out(i, c++) = w.log_cdf(y);
        out(i, c++) = w.log_ccdf(y);
        out(i, c++) = w.cdf(y);
        out(i, c++) = w.quantile(0.3);
        out(i, c++) = w.quantile_from_log(-5.0);
        out(i, c++) = w.quantile_from_log_upper(-5.0);
        out(i, c++) = w.scale();
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double sig[4] = {0.3, 1.0, 2.5, 12.0};
    const double s = sig[i % 4];
    const ld::weibull_distrib w{1.7, s};
    const double y = 0.2 + 0.25 * i;
    const auto d6 = w.logpdf_d6(y);
    const auto c6 = w.log_cdf_d6(y);
    const auto u6 = w.log_ccdf_d6(y);
    const auto j = w.expected_info_d4();
    std::vector<double> r;
    for (int b = 0; b <= 6; ++b) r.push_back(d6[b]);
    for (int b = 0; b <= 6; ++b) r.push_back(c6[b]);
    for (int b = 0; b <= 6; ++b) r.push_back(u6[b]);
    for (int b = 0; b <= 4; ++b) r.push_back(j.pair[1][b] + j.pair[2][b]);
    r.push_back(w.logpdf_dy_dk<2, 3>(y)[1]);
    r.push_back(w.logpdf(y));
    r.push_back(w.log_cdf(y));
    r.push_back(w.log_ccdf(y));
    r.push_back(w.cdf(y));
    r.push_back(w.quantile(0.3));
    r.push_back(w.quantile_from_log(-5.0));
    r.push_back(w.quantile_from_log_upper(-5.0));
    r.push_back(w.scale());
    // the order-six tensors are compared on a common scale per tensor: the host and the device differ in the
    // last bits of lgamma and polygamma, which the Bell sums amplify
    const int groups[4][2] = {{0, 7}, {7, 14}, {14, 21}, {21, 26}};
    for (const auto& g : groups) {
      double sc = 0.0;
      for (int c = g[0]; c < g[1]; ++c) sc = std::fmax(sc, std::fabs(r[c]));
      for (int c = g[0]; c < g[1]; ++c) {
        INFO("i " << i << " component " << c);
        CHECK(std::fabs(h(i, c) - r[c]) <= 1e-9 * sc);
      }
    }
    for (int c = 26; c < (int)r.size(); ++c) {
      INFO("i " << i << " component " << c);
      CHECK(h(i, c) == doctest::Approx(r[c]).epsilon(1e-10));
    }
  }
}

TEST_CASE("weibull: every order inside a kernel") { weibull_kernel_case(); }
