// vonmises_distrib (include/laplace/distrib/vonmises.hpp): the value and every derivative order against
// mpmath (tools/reference/distrib_vonmises.py, built from the definition: kappa a root of
// I_1/I_0 = exp(-s^2/2), derivatives by mpmath.diff and by reverting the series of A = I_1/I_0), evaluated
// inside a Kokkos kernel; the two limits (Gaussian as s -> 0, uniform as s -> infinity); the separation
// identity of the tensor; the score in its textbook form; periodicity; derivatives against differences;
// the expected information (closed forms, quadrature of the scores, the generic routes, the two limits);
// the cdf from -pi against quadrature, the quantile by round trips; the draw against Best and Fisher's
// rejection sampler; the start value; the order-six tensor on the host and in a kernel.
//
// Accuracy measure. A component is compared in ulp of its own value plus a floor of 4 eps times `mag`,
// the sum of the absolute values of the terms that add up to it (the reference file carries it): the
// components with a root of sin z or cos z, and every pure-s component where cos z = rho, are exact
// zeros of their formula, and a tolerance relative to the exact value fails there. For s above 4.5
// the conditioning of the map s -> kappa dominates (one eps of s moves kappa by s^2 eps) and the
// bound widens with s^2. The bounds below are provisional until measured on both builds;
// LAPLACE_PRINT_ERRORS=1 prints the worst values.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/vonmises.hpp>
#include <laplace/math/bessel_i.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included
using laplace_test::hide;

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

// ---------------------------------------------------------------------------------------------
// The device side, in named functions with external linkage (never a KOKKOS_LAMBDA in a TEST_CASE).

struct vonmises_query {
  double s, mu, y;
  int J, K, b;
};

// The component b of the tensor with J derivatives in y and K in the parameters (J = K = 0: the value).
LAPLACE_HD double vonmises_query_value(const ld::vonmises_distrib& d, double y, int J, int K, int b) {
#define VM_CASE(j, k) \
  case j * 10 + k: return d.logpdf_dy_dk<j, k>(y)[b];
  switch (J * 10 + K) {
    VM_CASE(0, 0)
    VM_CASE(0, 1) VM_CASE(0, 2) VM_CASE(0, 3) VM_CASE(0, 4) VM_CASE(0, 5) VM_CASE(0, 6)
    VM_CASE(1, 0) VM_CASE(1, 1) VM_CASE(1, 2) VM_CASE(1, 3) VM_CASE(1, 4) VM_CASE(1, 5)
    VM_CASE(2, 0) VM_CASE(2, 1) VM_CASE(2, 2) VM_CASE(2, 3) VM_CASE(2, 4)
    VM_CASE(3, 0) VM_CASE(3, 1) VM_CASE(3, 2) VM_CASE(3, 3)
    VM_CASE(4, 0) VM_CASE(4, 1) VM_CASE(4, 2)
    VM_CASE(5, 0) VM_CASE(5, 1)
    VM_CASE(6, 0)
    default: return 0.0;
  }
#undef VM_CASE
}

void vonmises_eval_batch(const std::vector<vonmises_query>& q, std::vector<double>& out) {
  const int n = int(q.size());
  Kokkos::View<double * [6]> in("vonmises_in", n);
  auto h = Kokkos::create_mirror_view(in);
  for (int i = 0; i < n; ++i) {
    h(i, 0) = q[i].s, h(i, 1) = q[i].mu, h(i, 2) = q[i].y;
    h(i, 3) = q[i].J, h(i, 4) = q[i].K, h(i, 5) = q[i].b;
  }
  Kokkos::deep_copy(in, h);
  Kokkos::View<double*> res("vonmises_res", n);
  Kokkos::parallel_for(
      "vonmises_batch", n, KOKKOS_LAMBDA(const int i) {
        const ld::vonmises_distrib d{in(i, 1), in(i, 0)};
        res(i) = vonmises_query_value(d, in(i, 2), int(in(i, 3)), int(in(i, 4)), int(in(i, 5)));
      });
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = hr(i);
}

// F, 1 - F, log F, log(1 - F) at (s, mu, y), four numbers per query, evaluated in a kernel
void vonmises_cdf_batch(const std::vector<vonmises_query>& q, std::vector<double>& out) {
  const int n = int(q.size());
  Kokkos::View<double * [3]> in("vonmises_cin", n);
  auto h = Kokkos::create_mirror_view(in);
  for (int i = 0; i < n; ++i) h(i, 0) = q[i].s, h(i, 1) = q[i].mu, h(i, 2) = q[i].y;
  Kokkos::deep_copy(in, h);
  Kokkos::View<double * [4]> res("vonmises_cres", n);
  Kokkos::parallel_for(
      "vonmises_cdf_batch", n, KOKKOS_LAMBDA(const int i) {
        const ld::vonmises_distrib d{in(i, 1), in(i, 0)};
        res(i, 0) = d.cdf(in(i, 2));
        res(i, 1) = d.ccdf(in(i, 2));
        res(i, 2) = d.log_cdf(in(i, 2));
        res(i, 3) = d.log_ccdf(in(i, 2));
      });
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  out.resize(4 * n);
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < 4; ++c) out[4 * i + c] = hr(i, c);
}

// The highest orders in one kernel, 40 numbers per point
LAPLACE_HD void vonmises_kernel_point(const ld::vonmises_distrib& d, double y, double* r) {
  int c = 0;
  const auto d6 = d.logpdf_d6(y);
  for (int b = 0; b <= 6; ++b) r[c++] = d6[b];
  const auto dy = d.logpdf_dy_dk<2, 3>(y);
  for (int b = 0; b <= 3; ++b) r[c++] = dy[b];
  const auto e4 = d.expected_info_d4();
  for (int b = 0; b <= 4; ++b) r[c++] = e4.pair[0][b] + e4.pair[2][b];
  const auto e0 = d.expected_info();
  r[c++] = e0[0];
  r[c++] = e0[2];
  r[c++] = d.logpdf(y);
  r[c++] = d.cdf(y);
  r[c++] = d.ccdf(y);
  r[c++] = d.log_cdf(y);
  r[c++] = d.log_ccdf(y);
  r[c++] = d.center();
  r[c++] = d.scale();
}
constexpr int vonmises_kernel_width = 7 + 4 + 5 + 2 + 1 + 4 + 2;

void vonmises_kernel_case() {
  const int n = 24;
  Kokkos::View<double * [vonmises_kernel_width]> out("vonmises_kernel_out", n);
  Kokkos::parallel_for(
      "distrib_vonmises", n, KOKKOS_LAMBDA(const int i) {
        const double sg[4] = {0.05, 0.6, 2.0, 5.0};
        const ld::vonmises_distrib d{0.7 * (i % 5) - 1.3, sg[i % 4]};
        double r[vonmises_kernel_width];
        vonmises_kernel_point(d, -3.0 + 0.25 * i, r);
        for (int c = 0; c < vonmises_kernel_width; ++c) out(i, c) = r[c];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double sg[4] = {0.05, 0.6, 2.0, 5.0};
    const ld::vonmises_distrib d{0.7 * (i % 5) - 1.3, sg[i % 4]};
    double r[vonmises_kernel_width];
    vonmises_kernel_point(d, -3.0 + 0.25 * i, r);
    // the order-six tensor and the others on a common scale per group: the host and the device differ in
    // the last bits of sin, cos and exp, which the sums amplify
    const int groups[4][2] = {{0, 7}, {7, 11}, {11, 16}, {16, 18}};
    for (const auto& g : groups) {
      double sc = 0.0;
      for (int c = g[0]; c < g[1]; ++c) sc = std::fmax(sc, std::fabs(r[c]));
      for (int c = g[0]; c < g[1]; ++c) {
        INFO("i " << i << " component " << c);
        CHECK(std::fabs(h(i, c) - r[c]) <= 1e-10 * sc);
      }
    }
    for (int c = 18; c < vonmises_kernel_width; ++c) {
      INFO("i " << i << " component " << c);
      CHECK(std::fabs(h(i, c) - r[c]) <= 1e-9 * std::fmax(std::fabs(r[c]), 1e-3));
    }
  }
}

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = std::numeric_limits<double>::epsilon();

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

// all the rows of a reference file as vectors of doubles (the header is skipped)
std::vector<std::vector<double>> read_table(const char* name) {
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[1024];
  std::vector<std::vector<double>> rows;
  if (!std::fgets(line, sizeof line, f)) return rows;
  while (std::fgets(line, sizeof line, f)) {
    std::vector<double> v;
    for (const auto& c : split_csv(line)) v.push_back(std::strtod(c.c_str(), nullptr));
    rows.push_back(v);
  }
  std::fclose(f);
  return rows;
}

double ulp_of(double v) { return v == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(v) - 52); }

// The bound, in units of the component's own scale, by order K and s.
//
// The unit is ulp_of(hi) + 4 eps mag, that is one ulp of the component plus the
// irreducible error of the cancellation that produced it, mag being the size of
// the terms that cancelled as the reference file records it. That sum, and not
// ulp_of(hi) alone, is what `scaled` below divides by, so it is what the bound
// must multiply: an earlier version multiplied only the ulp and added the floor
// unscaled, which made the asserted quantity different from the reported one.
// At s = 1e-8 the floor dominates the ulp by about five times, so the assertion
// allowed 0.399 where the reported measure was 4.15 — the two were not
// comparable, and a bound could not be read off the printed worst values.
//
// Measured on gcc over the whole reference grid, worst of the orders zero to
// six: 4.7, 5.0, 5.0, 5.1, 4.9, 4.9 and 5.0 for s up to 4.5, and 0.1 to 0.53
// above it. So one flat base of 8 covers every order with the usual margin of
// about half again, where the first guesses had run from 8 to 128. CUDA is
// still to measure and the base is to be taken as the worst of the two builds.
double ulp_bound(int K, double s) {
  static const double base[7] = {8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
  return base[K] * (s > 4.5 ? 1.0 + 0.25 * s * s : 1.0);
}

double worst_scaled[2][7] = {};

void check_tensor_file(const char* name, bool with_j) {
  const auto rows = read_table(name);
  CHECK(rows.size() > 5000);
  std::vector<vonmises_query> q;
  for (const auto& r : rows) {
    // logpdf: s,mu,y,K,b,hi,lo,mag ; dy: s,mu,y,J,K,b,hi,lo,mag
    vonmises_query x{r[0], r[1], r[2], 0, 0, 0};
    if (with_j) x.J = int(r[3]), x.K = int(r[4]), x.b = int(r[5]);
    else x.K = int(r[3]), x.b = int(r[4]);
    q.push_back(x);
  }
  std::vector<double> v;
  vonmises_eval_batch(q, v);
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    const int o = with_j ? 6 : 5;
    const double hi = r[o], lo = r[o + 1], mag = r[o + 2];
    const double err = std::fabs((v[i] - hi) - lo);
    const double floor_ = 4.0 * kEps * mag;
    const int order = q[i].J + q[i].K;
    const int region = q[i].s > 4.5 ? 1 : 0;
    const double scaled = err / (ulp_of(hi) + floor_);
    worst_scaled[region][order] = std::fmax(worst_scaled[region][order], scaled);
    INFO(name << ": s " << q[i].s << " mu " << q[i].mu << " y " << q[i].y << " J " << q[i].J << " K " << q[i].K << " b "
              << q[i].b << " value " << v[i] << " reference " << hi << " scaled " << scaled);
    CHECK(err <= ulp_bound(order, q[i].s) * (ulp_of(hi) + floor_));
  }
}

}  // namespace

TEST_CASE("vonmises: descriptors, support and kinks") {
  CHECK(ld::vonmises_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::vonmises_distrib::n_params == 2);
  CHECK(ld::vonmises_distrib::y_regularity == ld::infinitely_smooth);
  const auto& m = ld::vonmises_distrib::params[0];
  const auto& s = ld::vonmises_distrib::params[1];
  CHECK(m.lower == -ld::distrib_infinity);
  CHECK(m.upper == ld::distrib_infinity);
  CHECK(m.circular == true);
  CHECK(m.moves_support == false);
  CHECK(m.regularity == ld::infinitely_smooth);
  CHECK(s.lower == 0.0);
  CHECK(s.upper == ld::distrib_infinity);
  CHECK(s.circular == false);
  CHECK(s.moves_support == false);
  const ld::vonmises_distrib d{0.4, 0.9};
  CHECK(d.support_lower() == -kPi);
  CHECK(d.support_upper() == kPi);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == 0.9);
  // the centre is the mean direction wrapped, the support does not move with mu
  const ld::vonmises_distrib far{0.4 + 6.0 * kPi, 0.9};
  CHECK(far.center() == doctest::Approx(0.4).epsilon(1e-13));
  CHECK(far.support_lower() == -kPi);
  CHECK(d.param_kinks(0, 0.3).count == 0);
  CHECK(d.param_kinks(1, 0.3).count == 0);
  CHECK(d.y_kinks().count == 0);
}

TEST_CASE("vonmises: density against the textbook form exp(kappa cos z) / (2 pi I0(kappa))") {
  for (double s : {0.3, 1.0, 2.0}) {
    const double kappa = ld::vonmises_kappa_from_s(s);
    const ld::vonmises_distrib d{0.5, s};
    for (double z : {0.0, 0.7, 2.5, -3.0}) {
      const double f = std::exp(kappa * std::cos(z)) / (2.0 * kPi * std::cyl_bessel_i(0.0, kappa));
      INFO("s " << s << " z " << z);
      CHECK(std::exp(d.logpdf(0.5 + z)) == doctest::Approx(f).epsilon(1e-12));
    }
  }
}

TEST_CASE("vonmises: s -> 0 approaches the Gaussian at standard deviation s, with a correction of order s^2") {
  // The von Mises with the same rho is not the wrapped normal: kappa = 1/s^2 + 1/2 + ..., so
  // log f_vm - log f_normal = s^2 (-x^2/4 + x^4/24 + 1/8) + O(s^4) at z = s x. The wrapping correction
  // exp(-2 pi^2/s^2) is far below eps; this O(s^2) gap is what remains.
  for (double s : {1e-2, 1e-3}) {
    const ld::vonmises_distrib v{0.3, s};
    const ld::gaussian1_distrib g{0.3, s};
    for (double x : {0.0, 1.0, 2.0, -2.0}) {
      const double y = 0.3 + x * s;
      const double diff = v.logpdf(y) - g.logpdf(y);
      const double expect = s * s * (-0.25 * x * x + x * x * x * x / 24.0 + 0.125);
      INFO("s " << s << " x " << x << " difference " << diff << " expected " << expect);
      CHECK(std::fabs(diff - expect) <= 0.02 * s * s + 1e-14);
    }
    // the pure-mu second derivative, -kappa cos z against -1/s^2: relative gap s^2/2
    const double a = v.logpdf_d2(0.3)[0], b = g.logpdf_d2(0.3)[0];
    CHECK(std::fabs(a / b - 1.0) <= s * s);
    CHECK(std::fabs(a / b - 1.0 - 0.5 * s * s) <= 0.25 * s * s * s * s + 4e-15);
  }
}

TEST_CASE("vonmises: s -> infinity approaches the uniform density 1/(2 pi)") {
  for (double s : {6.0, 8.0}) {
    const double kappa = ld::vonmises_kappa_from_s(s);
    const ld::vonmises_distrib d{0.2, s};
    for (double y : {-3.0, 0.2, 2.0}) CHECK(std::fabs(d.logpdf(y) + std::log(2.0 * kPi)) <= 4.0 * kappa);
  }
  const ld::vonmises_distrib d{0.2, 20.0};
  for (double y : {-3.0, 0.2, 2.0}) CHECK(std::fabs(d.logpdf(y) + 1.8378770664093454836) <= 4.5e-16);
  CHECK(d.cdf(0.0) == doctest::Approx((0.0 + kPi) / (2.0 * kPi)).epsilon(1e-14));
}

TEST_CASE("vonmises: value and every derivative order against mpmath, in a kernel") {
  for (auto& w : worst_scaled) for (double& x : w) x = 0.0;
  check_tensor_file("vonmises_logpdf.csv", false);
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int r = 0; r < 2; ++r)
      for (int K = 0; K <= 6; ++K)
        std::printf("vonmises log f, order %d, s %s 4.5: worst error %.3g ulp of the component plus floor\n", K,
                    r ? "above" : "up to", worst_scaled[r][K]);
}

TEST_CASE("vonmises: mixed (y, theta) derivatives against mpmath, in a kernel") {
  for (auto& w : worst_scaled) for (double& x : w) x = 0.0;
  check_tensor_file("vonmises_dy.csv", true);
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int r = 0; r < 2; ++r)
      for (int K = 1; K <= 6; ++K)
        std::printf("vonmises mixed (y, theta), total order %d, s %s 4.5: worst error %.3g ulp of the component plus floor\n",
                    K, r ? "above" : "up to", worst_scaled[r][K]);
}

TEST_CASE("vonmises: separation, the component with a mu-derivatives is W_a(z) kappa^(b)(s)") {
  const double s = 0.7;
  double kap[7];
  ld::vonmises_kappa_from_s_upto<6>(s, kap);
  const double mu = 0.2;
  const ld::vonmises_distrib d{mu, s};
  const double y1 = 1.1, y2 = -0.8;
  for (int K = 1; K <= 6; ++K) {
    const auto t1 = lderiv(d, y1, K), t2 = lderiv(d, y2, K);
    for (int b = 0; b < K; ++b) {
      const int a = K - b;
      const double Wa[7] = {0.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0};  // the sin or cos factor of W_a
      const bool use_sin = (a % 4 == 1) || (a % 4 == 3);
      const double w1 = Wa[a] * (use_sin ? std::sin(y1 - mu) : std::cos(y1 - mu));
      const double w2 = Wa[a] * (use_sin ? std::sin(y2 - mu) : std::cos(y2 - mu));
      INFO("K " << K << " b " << b);
      CHECK(std::fabs(t1[b] - w1 * kap[b]) <= 16.0 * kEps * std::fabs(w1 * kap[b]));
      CHECK(std::fabs(t1[b] * w2 - t2[b] * w1) <= 32.0 * kEps * (std::fabs(t1[b] * w2) + std::fabs(t2[b] * w1)));
    }
  }
}

TEST_CASE("vonmises: the pure-s component against a Faa di Bruno twin through the Bessel-ratio derivatives") {
  namespace lm = laplace::math;
  for (double s : {0.5, 1.0, 2.0}) {
    double kap[7];
    ld::vonmises_kappa_from_s_upto<6>(s, kap);
    const double kappa = kap[0];
    const double fk[7] = {0.0, lm::bessel_i_ratio(kappa) - 1.0, lm::bessel_i_ratio_d1(kappa), lm::bessel_i_ratio_d2(kappa),
                          lm::bessel_i_ratio_d3(kappa), lm::bessel_i_ratio_d4(kappa), lm::bessel_i_ratio_d5(kappa)};
    const double z = 0.9;
    const double S = std::cos(z) - 1.0;
    const ld::vonmises_distrib d{0.0, s};
    for (int K = 1; K <= 6; ++K) {
      double Ld = 0.0;
      switch (K) {
        case 1: Ld = lm::chain_rule_d1(fk, kap); break;
        case 2: Ld = lm::chain_rule_d2(fk, kap); break;
        case 3: Ld = lm::chain_rule_d3(fk, kap); break;
        case 4: Ld = lm::chain_rule_d4(fk, kap); break;
        case 5: Ld = lm::chain_rule_d5(fk, kap); break;
        default: Ld = lm::chain_rule_d6(fk, kap); break;
      }
      const double twin = kap[K] * S - Ld;
      const double lib = lderiv(d, z, K)[K];
      INFO("s " << s << " K " << K);
      CHECK(std::fabs(lib - twin) <= 1e-9 * (std::fabs(kap[K] * S) + std::fabs(Ld)));
    }
  }
}

TEST_CASE("vonmises: the scores in their textbook forms") {
  for (double s : {0.05, 0.4, 1.3, 3.0}) {
    const double kappa = ld::vonmises_kappa_from_s(s);
    const double d1 = ld::vonmises_kappa_from_s_dk<1>(s);
    const double rho = std::exp(-0.5 * s * s);
    const ld::vonmises_distrib d{0.3, s};
    for (double y : {-2.0, 0.3, 1.0, 2.8}) {
      const auto t = d.logpdf_d1(y);
      INFO("s " << s << " y " << y);
      CHECK(std::fabs(t[0] - kappa * std::sin(y - 0.3)) <= 8.0 * kEps * kappa);
      CHECK(std::fabs(t[1] - d1 * (std::cos(y - 0.3) - rho)) <= 16.0 * kEps * std::fabs(d1) * (1.0 + rho));
    }
  }
}

TEST_CASE("vonmises: periodic in y and in mu, with the density wrapped") {
  // y -> y + 2 pi and mu -> mu + 2 pi change the rounding of y - mu by about eps (|y| + 2 pi), so the
  // results agree to that and not to the last bit; the floor is that perturbation times kappa^(b).
  for (double s : {0.1, 0.8, 2.5}) {
    double kap[7];
    ld::vonmises_kappa_from_s_upto<6>(s, kap);
    for (double mu : {0.4, -2.0, 3.0}) {
      for (double y : {-3.0, 0.1, 2.2}) {
        const ld::vonmises_distrib d0{mu, s}, d1{mu + 2.0 * kPi, s}, dm{mu - 4.0 * kPi, s};
        const double slack = 32.0 * kEps * (1.0 + std::fabs(y) + std::fabs(mu) + 12.0);
        CHECK(std::fabs(d0.logpdf(y + 2.0 * kPi) - d0.logpdf(y)) <= slack * kap[0] + 8.0 * kEps);
        CHECK(std::fabs(d1.logpdf(y) - d0.logpdf(y)) <= slack * kap[0] + 8.0 * kEps);
        CHECK(std::fabs(dm.logpdf(y) - d0.logpdf(y)) <= slack * kap[0] + 8.0 * kEps);
        for (int K = 1; K <= 6; ++K) {
          const auto a = lderiv(d0, y, K), b = lderiv(d0, y + 2.0 * kPi, K), c = lderiv(d1, y, K);
          for (size_t i = 0; i < a.size(); ++i) {
            // The floor is a magnitude, so kappa^(b) enters it in absolute
            // value: kappa falls as s grows, so its odd derivatives are
            // negative, and kappa'(0.1) is about -2000. Written without the
            // absolute value this floor came out at -4.38e-10 and no error,
            // however small, could meet it.
            const double fl = slack * std::fabs(kap[i]) + 32.0 * kEps * std::fabs(a[i]);
            const double extra = (i == size_t(K)) ? slack * std::fabs(kap[K]) : 0.0;
            INFO("s " << s << " mu " << mu << " y " << y << " K " << K << " component " << i);
            CHECK(std::fabs(a[i] - b[i]) <= fl + extra);
            CHECK(std::fabs(a[i] - c[i]) <= fl + extra);
          }
        }
      }
    }
  }
}

TEST_CASE("vonmises: derivatives against differences") {
  for (double s : {0.5, 1.5})
    for (double y : {-2.5, -1.0, 0.4, 1.7, 2.6}) check_derivs<ld::vonmises_distrib>({0.4, s}, y, 1e-7);
}

TEST_CASE("vonmises: mixed (y, theta) derivatives against differences") {
  for (double s : {0.5, 1.5})
    for (double y : {-1.2, 0.9}) check_y_derivs<ld::vonmises_distrib>({0.4, s}, y, 1e-3 * s, 1e-6);
  const ld::vonmises_distrib d{0.4, 0.8};
  CHECK(d.logpdf_dy_dk<1, 0>(2.0)[0] == doctest::Approx(-d.logpdf_dk<1>(2.0)[0]).epsilon(1e-14));
}

namespace {

// the closed information against the generic quadrature of the scores, orders 0, 1, 2
template <int K, class D>
double info_by_integral_error(const D& d, ld::info_form form) {
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
  std::vector<double> ex, gen;
  if constexpr (K == 0) ex = closed(d.expected_info());
  else ex = closed(d.template expected_info_dk<K>());
  if (form == ld::info_form::score) gen = closed(ld::expected_info_by_integral<K, ld::info_form::score>(hide(d)));
  else gen = closed(ld::expected_info_by_integral<K, ld::info_form::hessian>(hide(d)));
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

TEST_CASE("vonmises: expected information, diagonal exactly, against the quadrature of the scores") {
  for (double s : {0.3, 1.5, 4.0}) {
    for (double mu : {0.4, 3.0, -7.0}) {
      const ld::vonmises_distrib d{mu, s};
      const auto e = d.expected_info();
      const auto q = score_variance_continuous(d);
      INFO("s " << s << " mu " << mu);
      CHECK(e[1] == 0.0);
      CHECK(std::fabs(q[0] - e[0]) <= 1e-8 * e[0]);
      CHECK(std::fabs(q[2] - e[2]) <= 1e-8 * e[2]);
      CHECK(std::fabs(q[1]) <= 1e-8 * std::sqrt(e[0] * e[2]));
    }
  }
  // the closed form I_ss = s^2 rho^2 / A'(kappa) written with the Bessel-ratio derivative
  for (double s : {0.05, 0.5, 1.0, 2.0, 3.0}) {
    const double kappa = ld::vonmises_kappa_from_s(s);
    const double rho = std::exp(-0.5 * s * s);
    const ld::vonmises_distrib d{0.0, s};
    const auto e = d.expected_info();
    INFO("s " << s);
    CHECK(e[0] == doctest::Approx(kappa * rho).epsilon(1e-13));
    CHECK(e[2] == doctest::Approx(s * s * rho * rho / laplace::math::bessel_i_ratio_d1(kappa)).epsilon(1e-11));
  }
  // the limits: I_ss -> 2/s^2 as s -> 0 and I_ss -> 2 s^2 rho^2 as s -> infinity.
  //
  // The first limit is asserted with its own first correction, not bare. From
  // kappa = 1/s^2 + 1/2 + ..., rho^2 = 1 - s^2 + ... and
  // A'(kappa) = (s^4/2)(1 - s^2/2) + ..., the closed form gives
  //     I_ss s^2 / 2 = 1 - s^2/2 + O(s^4),
  // so the deviation from the bare limit is s^2/2 by itself: 5.0e-05 at
  // s = 1e-2, which is what an earlier version of this check reported against
  // a flat tolerance of 1e-5. Verified with mpmath at 40 digits: the residue
  // after subtracting 1 - s^2/2 is 8.3e-14, 8.3e-10 and 6.8e-08 at s = 1e-3,
  // 1e-2 and 3e-2, which is the O(s^4) the expansion predicts. Asserting the
  // rate rather than the limit is both sharper and the only correct form.
  for (double s : {1e-3, 1e-2, 3e-2}) {
    const auto e = ld::vonmises_distrib{0.0, s}.expected_info();
    const double v = e[2] * s * s / 2.0;
    INFO("s " << s << ": I_ss s^2/2 = " << v << " against 1 - s^2/2 = " << (1.0 - 0.5 * s * s));
    // What is left after 1 - s^2/2 is O(s^4): the residue measured in the
    // library divided by s^4 is 0.0836, 0.0834 and 0.0836 at the three s
    // below, constant over a factor of thirty in s and close to 1/12, and it
    // reproduces the mpmath residues (8.3e-14, 8.3e-10, 6.8e-08) to the digit.
    // The assertion bounds that coefficient rather than naming it, since three
    // points agreeing to three digits pin the rate but not a constant.
    CHECK(std::fabs(v - (1.0 - 0.5 * s * s)) <= 0.12 * s * s * s * s + 1e-16);
    CHECK(std::fabs(e[0] * s * s - 1.0) <= 1e-4);
  }
  for (double s : {6.0, 8.0}) {
    const double rho = std::exp(-0.5 * s * s);
    const auto e = ld::vonmises_distrib{0.0, s}.expected_info();
    CHECK(std::fabs(e[2] / (2.0 * s * s * rho * rho) - 1.0) <= 1e-12);
  }
}

TEST_CASE("vonmises: expected information against the generic routes, both forms") {
  for (double s : {0.5, 1.5, 3.0}) {
    const ld::vonmises_distrib d{0.4, s};
    INFO("s " << s);
    CHECK(info_by_integral_error<0>(d, ld::info_form::score) <= 1e-8);
    CHECK(info_by_integral_error<1>(d, ld::info_form::score) <= 1e-7);
    CHECK(info_by_integral_error<2>(d, ld::info_form::score) <= 1e-6);
    CHECK(info_by_integral_error<0>(d, ld::info_form::hessian) <= 1e-8);
    CHECK(info_by_integral_error<1>(d, ld::info_form::hessian) <= 1e-7);
  }
}

TEST_CASE("vonmises: derivatives of the expected information against differences and mpmath") {
  check_info_derivs<ld::vonmises_distrib>({0.4, 0.5}, 1e-6);
  check_info_derivs<ld::vonmises_distrib>({-2.0, 1.5}, 1e-6);
  const ld::vonmises_distrib d{0.4, 0.9};
  CHECK(d.expected_info_d1().pair[1][0] == 0.0);
  CHECK(d.expected_info_d4().pair[1][4] == 0.0);
  // the derivatives in mu vanish
  CHECK(d.expected_info_d3().pair[0][0] == 0.0);
  CHECK(d.expected_info_d3().pair[2][2] == 0.0);
  // mpmath: orders 1..4 of kappa rho and of s^2 rho^2 / A'
  for (const auto& r : read_table("vonmises_info_dk.csv")) {
    const ld::vonmises_distrib e{0.0, r[0]};
    const int K = int(r[1]);
    double v = 0.0;
    switch (K) {
      case 1: v = r[2] == 0 ? e.expected_info_d1().pair[0][1] : e.expected_info_d1().pair[2][1]; break;
      case 2: v = r[2] == 0 ? e.expected_info_d2().pair[0][2] : e.expected_info_d2().pair[2][2]; break;
      case 3: v = r[2] == 0 ? e.expected_info_d3().pair[0][3] : e.expected_info_d3().pair[2][3]; break;
      default: v = r[2] == 0 ? e.expected_info_d4().pair[0][4] : e.expected_info_d4().pair[2][4]; break;
    }
    const double ref = r[3] + r[4];
    INFO("s " << r[0] << " K " << K << " which " << r[2]);
    CHECK(std::fabs(v - ref) <= 2e-12 * std::fabs(ref) + 1e-300);
  }
  // mpmath: the quadrature of the squared scores
  for (const auto& r : read_table("vonmises_info.csv")) {
    const ld::vonmises_distrib e{0.0, r[0]};
    const auto t = e.expected_info();
    const double ref = r[2] + r[3];
    INFO("s " << r[0] << " which " << r[1]);
    if (r[1] == 1.0) CHECK(std::fabs(ref) <= 1e-12 * std::sqrt(t[0] * t[2]));
    else CHECK(std::fabs(t[int(r[1])] - ref) <= 1e-12 * std::fabs(ref));
  }
}

TEST_CASE("vonmises: the cdf from -pi against mpmath quadrature, in a kernel") {
  const auto rows = read_table("vonmises_cdf.csv");
  CHECK(rows.size() > 1000);
  std::map<std::tuple<double, double, double>, std::vector<double>> pts;
  for (const auto& r : rows) {
    auto& v = pts[std::make_tuple(r[0], r[1], r[2])];
    if (v.empty()) v.assign(4, 0.0);
    v[int(r[3])] = r[4] + r[5];
  }
  std::vector<vonmises_query> q;
  std::vector<std::vector<double>> refs;
  for (const auto& p : pts) {
    q.push_back({std::get<0>(p.first), std::get<1>(p.first), std::get<2>(p.first), 0, 0, 0});
    refs.push_back(p.second);
  }
  std::vector<double> v;
  vonmises_cdf_batch(q, v);
  double worst_series = 0.0, worst_tail = 0.0;
  for (size_t i = 0; i < q.size(); ++i) {
    const double kappa = ld::vonmises_kappa_from_s(q[i].s);
    const double m = std::fmin(refs[i][0], refs[i][1]);
    const bool series = kappa <= 3400.0 && m >= 1e-3;
    // The crossover of the arc mass's two routes, w0 sqrt(kappa) about one.
    //
    // Above kappa = 3400 the mass of an arc comes from a panel quadrature where
    // that converges and from the closed asymptotic form where it does not. The
    // two are at their best in opposite regimes and both are at their worst
    // where they meet, which is one Gaussian width from the mode: there the
    // quadrature reports non-convergence while the asymptotic sits at the
    // junction of its own two choices of the point at which g is evaluated.
    // Measured at s = 1e-4, where |y - mu| = s exactly and so w0 sqrt(kappa)
    // = 1: 2.9e-10 relative on F and 1.8e-09 on log F, against the 1e-15 the
    // family reaches everywhere else, at every mu and nowhere else on the grid.
    // The values themselves are right to ten digits (0.158655 and 0.841345,
    // which are Phi(-1) and Phi(1), the law being a Gaussian of deviation s
    // there).
    //
    // The exception is deliberately narrow, so that any other point losing
    // accuracy still fails. Closing it means a second order in g's expansion,
    // which is a choice about effort and not a repair; it is recorded as an
    // open corner in the numerics entry.
    const double w0rk = std::fabs(2.0 * std::sin(0.5 * (q[i].y - q[i].mu))) * std::sqrt(kappa);
    const bool crossover = !series && w0rk > 0.2 && w0rk < 5.0;
    for (int w = 0; w < 4; ++w) {
      const double p_ref = w % 2 == 0 ? refs[i][0] : refs[i][1];
      double tol_p = 5e-15 + (series ? 0.0 : 5e-10 * std::fmin(p_ref, m));
      if (crossover) tol_p = 5e-15 + 1e-8 * std::fmin(p_ref, m);
      const double err_p = std::fabs(v[4 * i + w] - refs[i][w]);
      const double tol = w < 2 ? tol_p : tol_p / p_ref;
      (series ? worst_series : worst_tail) = std::fmax(series ? worst_series : worst_tail, err_p / tol);
      INFO("s " << q[i].s << " mu " << q[i].mu << " y " << q[i].y << " which " << w << " value " << v[4 * i + w]
                << " reference " << refs[i][w]);
      CHECK(err_p <= tol);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("vonmises cdf: worst error over its tolerance, series %.3g, tail integral %.3g\n", worst_series, worst_tail);
}

TEST_CASE("vonmises: cdf identities, ends, monotonicity, continuity in mu, the density as its derivative") {
  for (double s : {0.01, 0.3, 1.0, 3.0}) {
    for (double mu : {0.3, 3.0, -3.1, 7.0}) {
      const ld::vonmises_distrib d{mu, s};
      CHECK(d.cdf(d.support_lower()) == 0.0);
      CHECK(d.cdf(d.support_upper()) == 1.0);
      CHECK(d.ccdf(d.support_lower()) == 1.0);
      CHECK(d.ccdf(d.support_upper()) == 0.0);
      CHECK(d.cdf(-10.0) == 0.0);
      CHECK(d.cdf(10.0) == 1.0);
      double prev = 0.0;
      for (int i = 1; i < 200; ++i) {
        const double y = -kPi + 2.0 * kPi * i / 200.0;
        const double F = d.cdf(y), G = d.ccdf(y);
        INFO("s " << s << " mu " << mu << " y " << y);
        CHECK(F >= prev - 1e-9 * std::fmax(F, 1e-3));
        CHECK(std::fabs(F + G - 1.0) <= 1e-13);
        prev = F;
      }
    }
  }
  // 2 pi-periodic and continuous in mu: the cut is in y, not in mu
  for (double y : {-3.0, -0.5, 1.0, 3.0}) {
    const double a = ld::vonmises_distrib{kPi - 1e-9, 0.4}.cdf(y);
    const double b = ld::vonmises_distrib{kPi + 1e-9, 0.4}.cdf(y);
    const double c = ld::vonmises_distrib{-kPi + 1e-9, 0.4}.cdf(y);
    INFO("y " << y);
    CHECK(std::fabs(a - b) <= 1e-8);
    CHECK(std::fabs(b - c) <= 1e-12);
  }
  // Not invariant under a rotation of the circle: F(y; mu) != F(y - c; mu - c)
  // in general, which is the real price of measuring F from a fixed -pi (the
  // cdf is analytic and 2 pi-periodic in mu, so there is no discontinuity to
  // find; this is what there is instead). The threshold is the measured size
  // of the effect and not a guess: with c = 1 and s = 0.5 the two differ by
  // 2.25e-04, and an independent mpmath quadrature gave 4.0e-05 at mu = 0 and
  // 1.5e-04 at mu = 1 for c = 0.4, so the effect is of order 1e-04 here. An
  // earlier version asked for more than 1e-3 and failed on a difference that
  // was real.
  {
    const double rot = std::fabs(ld::vonmises_distrib{0.0, 0.5}.cdf(-1.0) -
                                 ld::vonmises_distrib{1.0, 0.5}.cdf(0.0));
    INFO("rotation by 1 at s = 0.5 moves F by " << rot);
    CHECK(rot > 1e-5);
  }
  // dF/dy = f
  for (double s : {0.3, 1.0})
    for (double y : {-1.0, 0.5, 2.0}) {
      const ld::vonmises_distrib d{0.3, s};
      const double h = 1e-3;
      const double fd = (-d.cdf(y + 2 * h) + 8 * d.cdf(y + h) - 8 * d.cdf(y - h) + d.cdf(y - 2 * h)) / (12.0 * h);
      CHECK(fd == doctest::Approx(std::exp(d.logpdf(y))).epsilon(1e-6));
    }
}

TEST_CASE("vonmises: the generic derivatives of log F against differences of log F") {
  // the dispatchers take log_cdf_dk_by_integral: the family writes no closed form
  for (double s : {0.5, 1.2}) {
    const ld::vonmises_distrib d{0.3, s};
    for (double y : {-1.5, 0.9, 2.0}) {
      const auto t = ld::distrib_log_cdf_dk<1>(d, y);
      const double hm = 1e-4, hs = 1e-4 * s;
      const auto f = [&](double m, double ss) { return ld::vonmises_distrib{m, ss}.log_cdf(y); };
      const double dm = (-f(0.3 + 2 * hm, s) + 8 * f(0.3 + hm, s) - 8 * f(0.3 - hm, s) + f(0.3 - 2 * hm, s)) / (12 * hm);
      const double ds = (-f(0.3, s + 2 * hs) + 8 * f(0.3, s + hs) - 8 * f(0.3, s - hs) + f(0.3, s - 2 * hs)) / (12 * hs);
      INFO("s " << s << " y " << y);
      CHECK(std::fabs(t[0] - dm) <= 1e-6 * (1.0 + std::fabs(dm)));
      CHECK(std::fabs(t[1] - ds) <= 1e-6 * (1.0 + std::fabs(ds)));
    }
  }
}

TEST_CASE("vonmises: quantile by round trips, from p and from log p") {
  for (double s : {0.5, 0.01, 2.0}) {
    for (double mu : {0.3, 3.0, 7.0, -2.0}) {
      const ld::vonmises_distrib d{mu, s};
      for (double p : {1e-6, 0.01, 0.3, 0.5, 0.9, 1.0 - 1e-6}) {
        const double y = ld::distrib_quantile(d, p);
        INFO("s " << s << " mu " << mu << " p " << p << " quantile " << y);
        CHECK(y > -kPi - 1e-12);
        CHECK(y < kPi + 1e-12);
        if (p <= 0.5) CHECK(std::fabs(d.cdf(y) - p) <= 1e-9 * p + 1e-14);
        else CHECK(std::fabs(d.ccdf(y) - (1.0 - p)) <= 1e-9 * (1.0 - p) + 1e-14);
        const double yl = ld::distrib_quantile_from_log(d, std::log(p));
        CHECK(std::fabs(yl - y) <= 1e-7 * (1.0 + std::fabs(y)));
      }
    }
  }
}

namespace {

// Best and Fisher's wrapped-Cauchy rejection sampler for the von Mises law centred at 0 (kappa > 0)
double best_fisher(double kappa, std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double tau = 1.0 + std::sqrt(1.0 + 4.0 * kappa * kappa);
  const double rho = (tau - std::sqrt(2.0 * tau)) / (2.0 * kappa);
  const double r = (1.0 + rho * rho) / (2.0 * rho);
  for (;;) {
    const double u1 = u(g), u2 = u(g), u3 = u(g);
    const double z = std::cos(kPi * u1);
    const double f = (1.0 + r * z) / (r + z);
    const double c = kappa * (r - f);
    if (c * (2.0 - c) - u2 > 0.0 || std::log(c / u2) + 1.0 - c >= 0.0) return (u3 > 0.5 ? 1.0 : -1.0) * std::acos(f);
  }
}

double wrap_pi(double y) {
  while (y > kPi) y -= 2.0 * kPi;
  while (y <= -kPi) y += 2.0 * kPi;
  return y;
}

}  // namespace

TEST_CASE("vonmises: the cdf against the empirical cdf of Best and Fisher's sampler") {
  std::mt19937_64 g(20261115);
  for (double s : {0.3, 0.8}) {
    for (double mu : {2.5, -3.0}) {
      const ld::vonmises_distrib d{mu, s};
      const double kappa = ld::vonmises_kappa_from_s(s);
      const int n = 40000;
      std::vector<double> ys(n);
      for (int i = 0; i < n; ++i) ys[i] = wrap_pi(mu + best_fisher(kappa, g));
      for (double p : {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}) {
        int c = 0;
        for (double y : ys) c += y <= p;
        const double F = d.cdf(p), emp = double(c) / n;
        const double se = std::sqrt(F * (1.0 - F) / n);
        INFO("s " << s << " mu " << mu << " point " << p << " F " << F << " empirical " << emp);
        CHECK(std::fabs(emp - F) <= 4.0 * se + 1e-12);
      }
    }
  }
}

TEST_CASE("vonmises: the draw by inversion, reproducible and distributed as F") {
  const ld::vonmises_distrib d{1.0, 0.7};
  const int n = 2000;
  std::vector<double> ys(n);
  for (int i = 0; i < n; ++i) {
    ys[i] = ld::rng_by_inversion(d, 12345u, std::uint64_t(i));
    CHECK(ys[i] > -kPi - 1e-12);
    CHECK(ys[i] < kPi + 1e-12);
    // Reproducibility: the same (seed, index) must give the same number, which
    // is the property counter_uniform_source exists for. Asserted bitwise, but
    // reported in ulp when it fails, because one ulp between two textually
    // identical calls would be the compiler contracting them differently
    // rather than the draw being irreproducible, and the two call for opposite
    // repairs: a tolerance of a few ulp in the first case, a repair of the
    // draw's own chain in the second.
    if (i < 20) {
      const double again = ld::rng_by_inversion(d, 12345u, std::uint64_t(i));
      if (ys[i] != again) {
        const double m = std::fmax(std::fabs(ys[i]), 1e-300);
        const double u = std::nextafter(m, 1e308) - m;
        INFO("draw " << i << ": " << ys[i] << " then " << again << ", difference "
                     << std::fabs(ys[i] - again) / u << " ulp");
      }
      CHECK(ys[i] == again);
    }
  }
  for (double p : {-2.0, -0.5, 1.0, 2.5}) {
    int c = 0;
    for (double y : ys) c += y <= p;
    const double F = d.cdf(p), emp = double(c) / n;
    INFO("point " << p << " F " << F << " empirical " << emp);
    CHECK(std::fabs(emp - F) <= 4.0 * std::sqrt(F * (1.0 - F) / n) + 1e-12);
  }
}

TEST_CASE("vonmises: start_value from the sample") {
  const ld::vonmises_distrib proto{0.0, 1.0};
  // a concentrated sample about 0.8
  std::vector<double> y;
  for (int i = 0; i < 50; ++i) y.push_back(0.8 + 0.2 * std::sin(1.7 * i));
  double c = 0.0, sn = 0.0;
  for (double v : y) c += std::cos(v), sn += std::sin(v);
  c /= 50.0, sn /= 50.0;
  const auto st = ld::distrib_start_value(proto, y);
  CHECK(st.mu == doctest::Approx(std::atan2(sn, c)).epsilon(1e-14));
  CHECK(st.s == doctest::Approx(std::sqrt(-2.0 * std::log(std::sqrt(c * c + sn * sn)))).epsilon(1e-12));
  CHECK(st.mu == doctest::Approx(0.8).epsilon(0.05));
  // across the cut the estimate is still the mean direction (here about pi, represented near -pi or pi)
  std::vector<double> w = {kPi - 0.05, -kPi + 0.05, kPi - 0.1, -kPi + 0.1};
  const auto sw = ld::distrib_start_value(proto, w);
  CHECK(std::fabs(std::fabs(sw.mu) - kPi) <= 1e-12);
  CHECK(sw.s > 0.0);
  // no spread: s is floored at 1e-4; uniform ring: s is 4.29; empty and non-finite samples: (0, 1)
  const auto same = ld::distrib_start_value(proto, std::vector<double>(10, 0.5));
  CHECK(same.mu == doctest::Approx(0.5).epsilon(1e-14));
  CHECK(same.s == 1e-4);
  std::vector<double> ring;
  for (int i = 0; i < 8; ++i) ring.push_back(-kPi + 2.0 * kPi * i / 8.0);
  CHECK(ld::distrib_start_value(proto, ring).s == doctest::Approx(std::sqrt(-std::log(1e-8))).epsilon(1e-3));
  CHECK(ld::distrib_start_value(proto, std::vector<double>{}).s == 1.0);
  CHECK(ld::distrib_start_value(proto, std::vector<double>{std::nan(""), INFINITY}).mu == 0.0);
}

TEST_CASE("vonmises: every order inside a kernel") { vonmises_kernel_case(); }
