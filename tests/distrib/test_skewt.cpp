// skewt_distrib, Azzalini's skew t (include/laplace/distrib/skewt.hpp).
//
// What is checked, and against what:
//   * the value against the definition written out here (log 2, the Student log-density and
//     log T_(nu+1)(w), each from laplace::math);
//   * every component of the derivative tensor against nested central differences of the
//     value, which resolve the orders one to three;
//   * the mixed (y, theta) derivatives against the same differences in y;
//   * alpha = 0 against student_t1_distrib, where the family IS the Student: every order, in
//     the three parameters they share, is required to agree to the last bit;
//   * the value at nu = 1e8 against skewnormal_distrib at the parameters the direct-to-centred
//     map implies, the skew normal being the limit;
//   * the derivatives of the log-density against the mpmath reference of
//     tools/reference/distrib_skewt_ref.py, in ulp of the largest component of each order;
//   * the cdf, which this family does not write: the generic route of generic_cdf.hpp against
//     central differences of its own log F and against the quantile by root finding;
//   * the expected information, also generic, in its two forms against each other and against
//     the variance of the score, and at alpha = 0 against student_t1's closed information;
//   * the whole tensor inside a Kokkos kernel, host against device.
#include <doctest/doctest.h>

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/skewnormal.hpp>
#include <laplace/distrib/skewt.hpp>
#include <laplace/distrib/student_t1.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <type_traits>
#include <string>
#include <tuple>
#include <algorithm>
#include <vector>

namespace ld = laplace::distrib;
namespace lm = laplace::math;

namespace {

using ld::skewt_distrib;

// ---------------------------------------------------------------- helpers

template <int K>
int pos4(int a, int b, int c, int d) {
  return ld::detail_skewt::pos4<K>(a, b, c, d);
}

double ulp_of(double x) {
  const double a = std::fabs(x);
  return a > 0.0 ? std::nextafter(a, 1e308) - a : std::nextafter(0.0, 1.0);
}

// the definition, independent of the header's own assembly
double logpdf_definition(const skewt_distrib& d, double y) {
  const double z = (y - d.mu) / d.sigma;
  double cp[1];
  lm::own::detail_st::cpoly(d.nu, 0, cp);
  const double w = d.alpha * z * std::sqrt((d.nu + 1.0) / (d.nu + z * z));
  return std::log(2.0) + cp[0] - std::log(d.sigma) - 0.5 * (d.nu + 1.0) * std::log1p(z * z / d.nu) +
         lm::own::log_student_cdf(w, d.nu + 1.0);
}

// nested central difference of order n in the parameters named by idx (0 mu, 1 sigma, 2 alpha, 3 nu)
double fd(const int* idx, int n, double mu, double sigma, double alpha, double nu, double y) {
  if (n == 0) return skewt_distrib{mu, sigma, alpha, nu}.logpdf(y);
  const double base[4] = {std::fabs(mu) + 1.0, sigma, std::fabs(alpha) + 1.0, nu};
  const double h = 1e-3 * base[idx[0]];
  double up[4] = {mu, sigma, alpha, nu}, dn[4] = {mu, sigma, alpha, nu};
  up[idx[0]] += h;
  dn[idx[0]] -= h;
  return (fd(idx + 1, n - 1, up[0], up[1], up[2], up[3], y) -
          fd(idx + 1, n - 1, dn[0], dn[1], dn[2], dn[3], y)) /
         (2.0 * h);
}

// difference in y of the tensor of order K, for the mixed derivatives
template <int K>
double fd_y(const skewt_distrib& d, double y, int comp, int J) {
  if (J == 0) return d.logpdf_dk<K>(y)[comp];
  const double h = 1e-3 * d.sigma;
  return (fd_y<K>(d, y + h, comp, J - 1) - fd_y<K>(d, y - h, comp, J - 1)) / (2.0 * h);
}

// The error against the differences is measured against the largest component of the order, not
// against the component itself: an order has components that vanish exactly (the Student's own
// (mu^2, sigma) is one), where the difference is pure noise of the size of the order's scale.
template <int K>
void check_against_differences(const skewt_distrib& d, double y, double tol, double& worst) {
  const ld::sym_tensor<4, K> t = d.logpdf_dk<K>(y);
  double scale = 0.0;
  for (int i = 0; i < ld::sym_tensor<4, K>::size; ++i) scale = std::fmax(scale, std::fabs(t[i]));
  for (int b = 0; b <= K; ++b)
    for (int c = 0; b + c <= K; ++c)
      for (int e = 0; b + c + e <= K; ++e) {
        const int a = K - b - c - e;
        int idx[K];
        int p = 0;
        for (int i = 0; i < a; ++i) idx[p++] = 0;
        for (int i = 0; i < b; ++i) idx[p++] = 1;
        for (int i = 0; i < c; ++i) idx[p++] = 2;
        for (int i = 0; i < e; ++i) idx[p++] = 3;
        const double got = t[pos4<K>(a, b, c, e)];
        const double want = fd(idx, K, d.mu, d.sigma, d.alpha, d.nu, y);
        const double rel = std::fabs(got - want) / (std::fabs(want) + scale);
        worst = std::fmax(worst, rel);
        INFO("K " << K << " (a,b,c,d) " << a << b << c << e << " alpha " << d.alpha << " nu " << d.nu
                  << " y " << y << ": got " << got << ", differences " << want << ", scale " << scale);
        CHECK(rel <= tol);
      }
}

// the parameters of the centred skew normal that the direct (xi, omega, alpha) implies
ld::skewnormal_distrib centred_of_direct(double xi, double omega, double alpha) {
  const double b = std::sqrt(2.0 / 3.14159265358979323846);
  const double delta = alpha / std::sqrt(1.0 + alpha * alpha);
  const double bd = b * delta;
  const double v = 1.0 - bd * bd;
  const double mean = xi + omega * bd;
  const double sd = omega * std::sqrt(v);
  const double gamma = 0.5 * (4.0 - 3.14159265358979323846) * bd * bd * bd / (v * std::sqrt(v));
  return ld::skewnormal_distrib{mean, sd, gamma};
}

// ---------------------------------------------------------------- the reference

struct ref_row {
  int K = 0, a = 0, b = 0, c = 0, d = 0;
  double y = 0, hi = 0, lo = 0, mu = 0, sigma = 0, alpha = 0, nu = 0;
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

std::vector<ref_row> read_reference() {
  std::vector<ref_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/distrib_skewt_logpdf.csv";
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if (s.size() != 12) continue;
    auto num = [&](int i) { return std::strtod(s[i].c_str(), nullptr); };
    auto in = [&](int i) { return std::atoi(s[i].c_str()); };
    ref_row r;
    r.y = num(0), r.K = in(1), r.a = in(2), r.b = in(3), r.c = in(4), r.d = in(5);
    r.hi = num(6), r.lo = num(7), r.mu = num(8), r.sigma = num(9), r.alpha = num(10), r.nu = num(11);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

std::vector<double> lp_tensor(const ref_row& r, int K) {
  const skewt_distrib d{r.mu, r.sigma, r.alpha, r.nu};
  std::vector<double> v;
  auto copy = [&v](const auto& t) {
    using T = std::remove_cv_t<std::remove_reference_t<decltype(t)>>;
    v.assign(t.v, t.v + T::size);
  };
  switch (K) {
    case 1: copy(d.logpdf_dk<1>(r.y)); break;
    case 2: copy(d.logpdf_dk<2>(r.y)); break;
    case 3: copy(d.logpdf_dk<3>(r.y)); break;
    case 4: copy(d.logpdf_dk<4>(r.y)); break;
    case 5: copy(d.logpdf_dk<5>(r.y)); break;
    default: copy(d.logpdf_dk<6>(r.y)); break;
  }
  return v;
}

int pos4_dyn(int K, int a, int b, int c, int d) {
  switch (K) {
    case 1: return pos4<1>(a, b, c, d);
    case 2: return pos4<2>(a, b, c, d);
    case 3: return pos4<3>(a, b, c, d);
    case 4: return pos4<4>(a, b, c, d);
    case 5: return pos4<5>(a, b, c, d);
    default: return pos4<6>(a, b, c, d);
  }
}

// measured on gcc and on CUDA with LAPLACE_PRINT_ERRORS; the bound is 1.5 times the worst of
// the two builds, per order, in ulp of the largest component of that order at that point.
// Measured on gcc: 4, 9, 63, 99, 336 and 1.3e3 ulp. The worst points are the small shape
// (nu = 0.3) at order six, the far tail with |alpha| large at the orders four and five, and
// nu = 6 with |z| in the thousands at the orders two and three.
constexpr double BOUND_LOGPDF[7] = {0.0, 6.0, 14.0, 95.0, 150.0, 505.0, 1950.0};

}  // namespace

// ---------------------------------------------------------------- the cases

TEST_CASE("skewt: descriptors and kinks") {
  CHECK(skewt_distrib::n_params == 4);
  CHECK(skewt_distrib::kind == ld::distrib_kind::continuous);
  CHECK(skewt_distrib::y_regularity == ld::infinitely_smooth);
  const skewt_distrib d{0.5, 2.0, 1.0, 4.0};
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.5);
  CHECK(d.scale() == 2.0);
  for (int j = 0; j < 4; ++j) CHECK(d.param_kinks(j, 1.0).count == 0);
  CHECK(d.y_kinks().count == 0);
  for (int j = 0; j < 4; ++j) CHECK(skewt_distrib::params[j].regularity == ld::infinitely_smooth);
}

TEST_CASE("skewt: the value against the definition") {
  double worst = 0.0;
  for (double alpha : {0.0, 0.8, -2.5, 7.0})
    for (double nu : {0.4, 1.0, 3.5, 40.0})
      for (double z : {-40.0, -2.0, -0.3, 0.0, 1.1, 9.0}) {
        const skewt_distrib d{0.5, 1.7, alpha, nu};
        const double y = d.mu + d.sigma * z;
        const double a = d.logpdf(y), b = logpdf_definition(d, y);
        worst = std::fmax(worst, std::fabs(a - b) / (std::fabs(b) + 1e-12));
      }
  CHECK(worst <= 1e-14);
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("skewt value against the definition: %.3g\n", worst);
}

TEST_CASE("skewt: the derivatives against central differences") {
  // the nested differences carry h^2 = 1e-6 per level, so they resolve the first three orders
  double w1 = 0.0, w2 = 0.0, w3 = 0.0;
  for (double alpha : {0.0, 0.7, -2.0, 5.0})
    for (double nu : {0.8, 3.0, 12.0})
      for (double z : {-2.0, -0.3, 1.5}) {
        const skewt_distrib d{0.3, 1.4, alpha, nu};
        const double y = d.mu + d.sigma * z;
        // measured 2.0e-6, 1.1e-5 and 6.7e-5 on gcc; the bound is a few times that, since what
        // it limits is the resolution of the differences and not the library's own error
        check_against_differences<1>(d, y, 1e-5, w1);
        check_against_differences<2>(d, y, 6e-5, w2);
        check_against_differences<3>(d, y, 4e-4, w3);
      }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("skewt against differences: order 1 %.3g, order 2 %.3g, order 3 %.3g\n", w1, w2, w3);
}

TEST_CASE("skewt: the mixed (y, theta) derivatives") {
  double worst = 0.0;
  for (double alpha : {0.0, 1.3, -3.0})
    for (double nu : {1.5, 6.0}) {
      const skewt_distrib d{0.3, 1.4, alpha, nu};
      const double y = d.mu + 0.8 * d.sigma;
      for (int J = 1; J <= 2; ++J) {
        const ld::sym_tensor<4, 2> t = J == 1 ? d.logpdf_dy_dk<1, 2>(y) : d.logpdf_dy_dk<2, 2>(y);
        double scale = 0.0;
        for (int i = 0; i < ld::sym_tensor<4, 2>::size; ++i) scale = std::fmax(scale, std::fabs(t[i]));
        for (int i = 0; i < ld::sym_tensor<4, 2>::size; ++i) {
          const double want = fd_y<2>(d, y, i, J);
          const double rel = std::fabs(t[i] - want) / (std::fabs(want) + scale);
          worst = std::fmax(worst, rel);
          INFO("J " << J << " component " << i << " alpha " << alpha << " nu " << nu);
          CHECK(rel <= 1e-4);  // measured 1.1e-5
        }
      }
    }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("skewt mixed (y, theta): %.3g\n", worst);
}

TEST_CASE("skewt at alpha = 0 is student_t1, to the last bit") {
  for (double nu : {0.4, 0.8, 1.0, 3.0, 12.0, 1e4})
    for (double z : {-40.0, -4.0, -0.3, 0.0, 1.9, 30.0}) {
      const skewt_distrib s{0.3, 1.4, 0.0, nu};
      const ld::student_t1_distrib r{0.3, 1.4, nu};
      const double y = s.mu + s.sigma * z;
      CHECK(s.logpdf(y) == doctest::Approx(r.logpdf(y)).epsilon(1e-15));
      // every order, in the three parameters the two families share
      const auto a1 = s.logpdf_dk<1>(y), b1b = s.logpdf_dk<1>(y);
      (void)b1b;
      const auto s6 = s.logpdf_dk<6>(y);
      const auto r6 = r.logpdf_dk<6>(y);
      for (int b = 0; b <= 6; ++b)
        for (int e = 0; b + e <= 6; ++e) {
          const int a = 6 - b - e;
          int idx3[6];
          int p = 0;
          for (int i = 0; i < a; ++i) idx3[p++] = 0;
          for (int i = 0; i < b; ++i) idx3[p++] = 1;
          for (int i = 0; i < e; ++i) idx3[p++] = 2;
          INFO("nu " << nu << " z " << z << " (a,b,nu) " << a << b << e);
          CHECK(s6[pos4<6>(a, b, 0, e)] == r6[ld::sym_tensor_position<3, 6>(idx3)]);
        }
      const auto s1 = s.logpdf_dk<1>(y);
      const auto r1 = r.logpdf_dk<1>(y);
      for (int b = 0; b <= 1; ++b)
        for (int e = 0; b + e <= 1; ++e) {
          const int a = 1 - b - e;
          int idx3[1];
          int p = 0;
          for (int i = 0; i < a; ++i) idx3[p++] = 0;
          for (int i = 0; i < b; ++i) idx3[p++] = 1;
          for (int i = 0; i < e; ++i) idx3[p++] = 2;
          CHECK(s1[pos4<1>(a, b, 0, e)] == r1[ld::sym_tensor_position<3, 1>(idx3)]);
        }
      (void)a1;
    }
}

TEST_CASE("skewt at large nu is the skew normal") {
  // the limit is approached like 1/nu, so the agreement at nu = 1e8 is of that size and the
  // test is that it tightens by the same factor from 1e6 to 1e8
  double worst[2] = {};
  int k = 0;
  for (double nu : {1e6, 1e8}) {
    for (double alpha : {0.5, 2.0, -3.0})
      for (double z : {-2.0, -0.3, 0.0, 1.4}) {
        const skewt_distrib d{0.4, 1.3, alpha, nu};
        const double y = d.mu + d.sigma * z;
        const auto sn = centred_of_direct(d.mu, d.sigma, alpha);
        const double rel = std::fabs(d.logpdf(y) - sn.logpdf(y)) / (std::fabs(sn.logpdf(y)) + 1e-12);
        worst[k] = std::fmax(worst[k], rel);
      }
    ++k;
  }
  CHECK(worst[0] <= 1e-5);
  CHECK(worst[1] <= 1e-7);
  CHECK(worst[1] < worst[0]);
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("skewt against the skew normal: nu = 1e6 %.3g, nu = 1e8 %.3g\n", worst[0], worst[1]);
}

TEST_CASE("skewt: the generic cdf and quantile") {
  // this family writes no cdf member: the dispatchers quadrature the density
  for (double alpha : {0.0, 1.5, -2.0})
    for (double nu : {1.0, 4.0}) {
      const skewt_distrib d{0.2, 1.1, alpha, nu};
      for (double z : {-2.0, -0.4, 0.7, 2.5}) {
        const double y = d.mu + d.sigma * z;
        const double lf = ld::distrib_log_cdf(d, y), lc = ld::distrib_log_ccdf(d, y);
        INFO("alpha " << alpha << " nu " << nu << " z " << z);
        CHECK(std::isfinite(lf));
        CHECK(std::isfinite(lc));
        // F + (1 - F) = 1
        CHECK(std::exp(lf) + std::exp(lc) == doctest::Approx(1.0).epsilon(1e-12));
        // the derivative of F in y is the density
        const double h = 1e-4 * d.sigma;
        const double dF = (ld::distrib_cdf(d, y + h) - ld::distrib_cdf(d, y - h)) / (2.0 * h);
        CHECK(dF == doctest::Approx(std::exp(d.logpdf(y))).epsilon(1e-6));
        // the quantile inverts it
        const double p = std::exp(lf);
        if (p > 1e-10 && p < 1.0 - 1e-10)
          CHECK(ld::distrib_quantile(d, p) == doctest::Approx(y).epsilon(1e-7));
      }
    }
}

TEST_CASE("skewt: the expected information by the generic quadrature") {
  for (double alpha : {0.0, 1.0, -2.0})
    for (double nu : {3.0, 9.0}) {
      const skewt_distrib d{0.2, 1.3, alpha, nu};
      const auto h = ld::expected_info_by_integral<0, ld::info_form::hessian>(d);
      const auto s = ld::expected_info_by_integral<0, ld::info_form::score>(d);
      double scale = 0.0;
      for (int i = 0; i < ld::sym_tensor<4, 2>::size; ++i) scale = std::fmax(scale, std::fabs(h[i]));
      for (int i = 0; i < ld::sym_tensor<4, 2>::size; ++i) {
        INFO("alpha " << alpha << " nu " << nu << " component " << i);
        CHECK(std::fabs(h[i] - s[i]) <= 1e-7 * scale);
      }
      // at alpha = 0 the Student's closed information, in the three shared parameters
      if (alpha == 0.0) {
        const ld::student_t1_distrib r{d.mu, d.sigma, d.nu};
        const auto ri = r.expected_info();
        const int map3[6][2] = {{0, 0}, {0, 1}, {0, 3}, {1, 1}, {1, 3}, {3, 3}};
        for (int q = 0; q < 6; ++q) {
          int i4[2] = {map3[q][0], map3[q][1]};
          const double got = h[ld::sym_tensor_position<4, 2>(i4)];
          int i3[2] = {map3[q][0] == 3 ? 2 : map3[q][0], map3[q][1] == 3 ? 2 : map3[q][1]};
          const double want = ri[ld::sym_tensor_position<3, 2>(i3)];
          INFO("nu " << nu << " pair " << q);
          CHECK(std::fabs(got - want) <= 1e-7 * std::fmax(std::fabs(want), scale));
        }
      }
    }
}

TEST_CASE("skewt: the log-density derivatives against mpmath") {
  // tools/reference/distrib_skewt_ref.py: SymPy's own chain rule over the definition, with the
  // derivatives of log T_(nu+1) by quadrature of the density's nu-derivatives, at 60 and 90
  // digits. Each component is compared in ulp of the largest component of its order at that
  // point, the components changing sign within an order.
  const auto rows = read_reference();
  CHECK(rows.size() > 5000);
  std::map<std::tuple<double, double, double, double, double, int>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.y, r.mu, r.sigma, r.alpha, r.nu, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  std::map<std::tuple<double, double, double, double, double, int>, std::vector<double>> cache;
  double worst[7] = {};
  std::tuple<double, double, double, int, int, int> where[7];
  std::map<double, double> by_nu, by_alpha;
  std::map<double, double> by_z_large_nu;  // the rows with nu >= 1000, by z
  for (const auto& r : rows) {
    const auto key = std::make_tuple(r.y, r.mu, r.sigma, r.alpha, r.nu, r.K);
    auto it = cache.find(key);
    if (it == cache.end()) {
      if (cache.size() > 64) cache.clear();
      it = cache.emplace(key, lp_tensor(r, r.K)).first;
    }
    const double v = it->second[pos4_dyn(r.K, r.a, r.b, r.c, r.d)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double u = err / ulp_of(scale[key]);
    by_nu[r.nu] = std::fmax(by_nu[r.nu], u);
    by_alpha[r.alpha] = std::fmax(by_alpha[r.alpha], u);
    if (r.nu >= 1000.0) {
      const double z = (r.y - r.mu) / r.sigma;
      by_z_large_nu[z] = std::fmax(by_z_large_nu[z], u);
    }
    if (u > worst[r.K]) {
      worst[r.K] = u;
      where[r.K] = std::make_tuple(r.y, r.alpha, r.nu, r.b, r.c, r.d);
    }
    INFO("y " << r.y << " K " << r.K << " (a,b,c,d) " << r.a << r.b << r.c << r.d << " mu " << r.mu
              << " sigma " << r.sigma << " alpha " << r.alpha << " nu " << r.nu << " ulp " << u);
    CHECK(u <= BOUND_LOGPDF[r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (int K = 1; K <= 6; ++K)
      std::printf("skewt logpdf, order %d: max %.3g ulp of the order's scale (y %.6g alpha %.4g nu %.8g, b %d c %d d %d)\n",
                  K, worst[K], std::get<0>(where[K]), std::get<1>(where[K]), std::get<2>(where[K]),
                  std::get<3>(where[K]), std::get<4>(where[K]), std::get<5>(where[K]));
    std::printf("skewt logpdf by nu:");
    for (const auto& kv : by_nu) std::printf(" %.6g:%.3g", kv.first, kv.second);
    std::printf("\nskewt logpdf by alpha:");
    for (const auto& kv : by_alpha) std::printf(" %.6g:%.3g", kv.first, kv.second);
    std::printf("%s", "\n");
  }
}

// The kernel lives in a named function with external linkage, never in the body of a TEST_CASE:
// nvcc tags an extended device lambda with the address of its enclosing function, and doctest's
// case function is a static DOCTEST_ANON_FUNC_<counter> whose counter restarts in every
// translation unit (CLAUDE.md, the traps of 2026-11-13).
void skewt_kernel_case();

void skewt_kernel_case() {
  constexpr int n = 12;
  Kokkos::View<double*> dev("dev", n * ld::sym_tensor<4, 6>::size);
  Kokkos::parallel_for(
      "skewt_kernel", n, KOKKOS_LAMBDA(const int i) {
        const double alpha = -3.0 + 1.0 * i;
        const skewt_distrib d{0.3, 1.2, alpha, 2.0 + 0.5 * i};
        const auto t = d.logpdf_dk<6>(0.9);
        for (int c = 0; c < ld::sym_tensor<4, 6>::size; ++c) dev(i * ld::sym_tensor<4, 6>::size + c) = t[c];
      });
  auto host = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host, dev);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double alpha = -3.0 + 1.0 * i;
    const skewt_distrib d{0.3, 1.2, alpha, 2.0 + 0.5 * i};
    const auto t = d.logpdf_dk<6>(0.9);
    double scale = 0.0;
    for (int c = 0; c < ld::sym_tensor<4, 6>::size; ++c) scale = std::fmax(scale, std::fabs(t[c]));
    for (int c = 0; c < ld::sym_tensor<4, 6>::size; ++c)
      worst = std::fmax(worst, std::fabs(host(i * ld::sym_tensor<4, 6>::size + c) - t[c]) / scale);
  }
  INFO("host against device, order 6, relative to the largest component: " << worst);
  CHECK(worst <= 1e-12);
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("skewt kernel host/device: %.3g\n", worst);
}

TEST_CASE("skewt: every order inside a kernel") { skewt_kernel_case(); }

// The generic cdf of a four-parameter family inside a kernel: the route keeps the integrand's
// derivative tensors on the stack at every node of the quadrature, so this is what sets the
// per-thread stack the CUDA build asks for (tests/main.cpp).
void skewt_generic_cdf_kernel_case();

void skewt_generic_cdf_kernel_case() {
  constexpr int n = 8;
  Kokkos::View<double*> dev("dev", n * (1 + ld::sym_tensor<4, 1>::size));
  Kokkos::parallel_for(
      "skewt_generic_cdf_kernel", n, KOKKOS_LAMBDA(const int i) {
        const skewt_distrib d{0.2, 1.1, -1.0 + 0.5 * i, 2.0 + i};
        const double y = d.mu + 0.6 * d.sigma;
        dev(i * (1 + ld::sym_tensor<4, 1>::size)) = ld::distrib_log_cdf(d, y);
        const auto t = ld::distrib_log_cdf_dk<1>(d, y);
        for (int c = 0; c < ld::sym_tensor<4, 1>::size; ++c)
          dev(i * (1 + ld::sym_tensor<4, 1>::size) + 1 + c) = t[c];
      });
  auto host = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host, dev);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const skewt_distrib d{0.2, 1.1, -1.0 + 0.5 * i, 2.0 + i};
    const double y = d.mu + 0.6 * d.sigma;
    const double lf = ld::distrib_log_cdf(d, y);
    const auto t = ld::distrib_log_cdf_dk<1>(d, y);
    double scale = std::fabs(lf);
    for (int c = 0; c < ld::sym_tensor<4, 1>::size; ++c) scale = std::fmax(scale, std::fabs(t[c]));
    worst = std::fmax(worst, std::fabs(host(i * (1 + ld::sym_tensor<4, 1>::size)) - lf) / scale);
    for (int c = 0; c < ld::sym_tensor<4, 1>::size; ++c)
      worst = std::fmax(worst, std::fabs(host(i * (1 + ld::sym_tensor<4, 1>::size) + 1 + c) - t[c]) / scale);
  }
  INFO("the generic cdf, host against device, relative to the largest component: " << worst);
  CHECK(worst <= 1e-9);
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("skewt generic cdf host/device: %.3g\n", worst);
}

TEST_CASE("skewt: the generic cdf inside a kernel") { skewt_generic_cdf_kernel_case(); }

TEST_CASE("skewt: the draw from the exact representation") {
  // the family writes its own draw (two normals and one gamma, each by inversion) instead of
  // inverting the cdf, which the generic route would do at 5.8 ms a draw against 529 ns here
  const int n = 40000;
  for (double alpha : {0.0, 1.5, -3.0})
    for (double nu : {2.5, 8.0}) {
      const skewt_distrib d{0.3, 1.2, alpha, nu};
      std::vector<double> v(n);
      for (int i = 0; i < n; ++i) v[i] = d.rng(20261115u, i);
      // the same observation gives the same number again
      CHECK(d.rng(20261115u, 7) == v[7]);
      std::sort(v.begin(), v.end());
      // the empirical cdf against the generic cdf, within four standard errors
      for (double z : {-2.0, -0.5, 0.4, 1.5}) {
        const double y = d.mu + d.sigma * z;
        const double emp = double(std::lower_bound(v.begin(), v.end(), y) - v.begin()) / n;
        const double ex = ld::distrib_cdf(d, y);
        const double se = std::sqrt(ex * (1.0 - ex) / n);
        INFO("alpha " << alpha << " nu " << nu << " z " << z << ": empirical " << emp << ", cdf " << ex);
        CHECK(std::fabs(emp - ex) <= 4.0 * se + 1e-4);
      }
      // the mean against its closed form, where it exists (nu > 1)
      const double delta = alpha / std::sqrt(1.0 + alpha * alpha);
      const double b = std::sqrt(nu / 3.14159265358979323846) *
                       std::exp(std::lgamma(0.5 * (nu - 1.0)) - std::lgamma(0.5 * nu));
      double m = 0.0;
      for (double x : v) m += x;
      m /= n;
      const double mean = d.mu + d.sigma * delta * b;
      // the standard error of the sample mean, from the closed variance (nu > 2 here)
      const double var = d.sigma * d.sigma * (nu / (nu - 2.0) - delta * delta * b * b);
      INFO("alpha " << alpha << " nu " << nu << ": sample mean " << m << ", mean " << mean);
      CHECK(std::fabs(m - mean) <= 5.0 * std::sqrt(var / n));
    }
}

TEST_CASE("skewt: start values") {
  ld::distrib_sample_summary s{};
  s.count = 100.0;
  s.median = 2.5;
  s.lower_quartile = 1.0;
  s.upper_quartile = 4.0;
  s.skewness = 0.8;
  const skewt_distrib d = skewt_distrib{0.0, 1.0, 0.0, 1.0}.start_value(s);
  CHECK(d.mu == doctest::Approx(2.5));
  CHECK(d.sigma == doctest::Approx(1.5));
  CHECK(d.alpha == doctest::Approx(1.0));
  CHECK(d.nu == doctest::Approx(5.0));
  s.skewness = -0.3;
  const skewt_distrib e = skewt_distrib{0.0, 1.0, 0.0, 1.0}.start_value(s);
  CHECK(e.alpha == doctest::Approx(-1.0));
}
