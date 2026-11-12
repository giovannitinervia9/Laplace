// Partial derivatives of log P(a, a c) and log Q(a, a c) in a (at fixed c) and
// c, total orders 1 to 6 (include/laplace/math/incomplete_gamma_ratio.hpp),
// and the derivatives in x of log P and log Q at integer a near the mean
// (incomplete_gamma_mixed.hpp, the Poisson cdf), against the mpmath references
// of tools/reference/incomplete_ratio.py, evaluated in a Kokkos kernel on the
// default execution space.
//
// Figures. Ratio files: the error of v_j = d^K / da^(K-j) dc^j in ulp of the
// largest component of its order at that point in the logarithmic variables,
// max_j' |a^(K-j') c^j' v_j'| / (a^(K-j) c^j): these are, up to their sign,
// the derivatives in (log phi, log mu) of the gamma distribution's log F, and
// the components change sign, so their own relative error is undefined at
// their zeros. Poisson files: the error of each component in ulp of the
// largest |value| of that component over the x of the same a (x within 4
// sqrt(a) of a). Bounds: the maxima measured on 2026-11-12 (gcc/OpenMP),
// half again. Ratio, P orders 1-6: 1496, 2113, 2082, 1736, 2050, 2638;
// Q: 1704, 1432, 1204, 1179, 1701, 1892 (orders 5-6 were 9.0e3, 7.3e4 and
// 5.8e4, 6.2e4 before the repairs of the night of 2026-11-12 to the pure
// derivatives in c near c = 1 at large a, docs/numerics.md). The figures of
// orders 1-4 are complements log(1 - S) with S near 1e-250, where the error
// of exp(log S) (|log S| eps relative, as in ulpL) dominates; those of orders
// 5-6 are derivatives with five or six in c below a = 50 near c = 1 and at
// a = 13097, c = 1.33. Poisson, P orders 1-6: 4.73, 12.6, 112, 467, 938,
// 4128 (order 5: 1536 at a = 12 on the CUDA build); Q: 4.00, 14.2, 72.5,
// 318, 715, 1449; after the repairs, with 78 points added at a = 300 to
// 1e5 (gcc): P 4.73, 9.04, 79.9, 370, 849, 4128; Q 4.00, 14.2, 62.4, 174,
// 715, 1425 (the bounds were kept).
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/incomplete_gamma_mixed.hpp>
#include <laplace/math/incomplete_gamma_ratio.hpp>

#include <limits>
#include <map>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;

struct Ratio {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double c, double jd) const {
    namespace lm = laplace::math;  // the wrappers
    const int j = static_cast<int>(jd);
    if (p) {
      switch (k) {
        case 1: return lm::log_gamma_p_ratio_mixed_d1(a, c).v[j];
        case 2: return lm::log_gamma_p_ratio_mixed_d2(a, c).v[j];
        case 3: return lm::log_gamma_p_ratio_mixed_d3(a, c).v[j];
        case 4: return lm::log_gamma_p_ratio_mixed_d4(a, c).v[j];
        case 5: return lm::log_gamma_p_ratio_mixed_d5(a, c).v[j];
        default: return lm::log_gamma_p_ratio_mixed_d6(a, c).v[j];
      }
    }
    switch (k) {
      case 1: return lm::log_gamma_q_ratio_mixed_d1(a, c).v[j];
      case 2: return lm::log_gamma_q_ratio_mixed_d2(a, c).v[j];
      case 3: return lm::log_gamma_q_ratio_mixed_d3(a, c).v[j];
      case 4: return lm::log_gamma_q_ratio_mixed_d4(a, c).v[j];
      case 5: return lm::log_gamma_q_ratio_mixed_d5(a, c).v[j];
      default: return lm::log_gamma_q_ratio_mixed_d6(a, c).v[j];
    }
  }
};

struct Mixed {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x, double jd) const {
    namespace lm = laplace::math;
    const int j = static_cast<int>(jd);
    if (p) {
      switch (k) {
        case 1: return lm::log_gamma_p_mixed_d1(a, x).v[j];
        case 2: return lm::log_gamma_p_mixed_d2(a, x).v[j];
        case 3: return lm::log_gamma_p_mixed_d3(a, x).v[j];
        case 4: return lm::log_gamma_p_mixed_d4(a, x).v[j];
        case 5: return lm::log_gamma_p_mixed_d5(a, x).v[j];
        default: return lm::log_gamma_p_mixed_d6(a, x).v[j];
      }
    }
    switch (k) {
      case 1: return lm::log_gamma_q_mixed_d1(a, x).v[j];
      case 2: return lm::log_gamma_q_mixed_d2(a, x).v[j];
      case 3: return lm::log_gamma_q_mixed_d3(a, x).v[j];
      case 4: return lm::log_gamma_q_mixed_d4(a, x).v[j];
      case 5: return lm::log_gamma_q_mixed_d5(a, x).v[j];
      default: return lm::log_gamma_q_mixed_d6(a, x).v[j];
    }
  }
};

double ulp_of(double s) { return std::ldexp(1.0, std::ilogb(std::max(s, 2.2250738585072014e-308)) - 52); }

void check_ratio(int k, bool p, double bound) {
  const std::string name = std::string(p ? "log_gamma_p_ratio_mixed_d" : "log_gamma_q_ratio_mixed_d") + std::to_string(k);
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, Ratio{k, p});
  // the scale of each point: max_j |a^(k-j) c^j v_j|
  std::map<std::pair<double, double>, double> scale;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double a = r.x[i][0], c = r.x[i][1];
    const int j = static_cast<int>(r.x[i][2]);
    const double u = std::fabs(r.hi[i]) * std::pow(a, k - j) * std::pow(c, j);
    auto& s = scale[{a, c}];
    s = std::max(s, u);
  }
  double worst = 0;
  bench::Args at{};
  std::size_t bad = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double a = r.x[i][0], c = r.x[i][1];
    const int j = static_cast<int>(r.x[i][2]);
    const double s = scale[{a, c}] / (std::pow(a, k - j) * std::pow(c, j));
    const double e = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / ulp_of(s);
    if (!std::isfinite(e)) {
      ++bad;
      continue;
    }
    if (e > worst) worst = e, at = r.x[i];
  }
  INFO(name, ": max ulp of the order's scale ", worst, " at (a ", at[0], ", c ", at[1], ", j = ", at[2], ")");
  CHECK(bad == 0);
  CHECK(worst <= bound);
}

void check_poisson(int k, bool p, double bound) {
  const std::string name = std::string(p ? "log_gamma_p_mixed_poisson_d" : "log_gamma_q_mixed_poisson_d") + std::to_string(k);
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, Mixed{k, p});
  std::map<std::pair<double, double>, double> scale;  // (a, j)
  for (std::size_t i = 0; i < y.size(); ++i) {
    auto& s = scale[{r.x[i][0], r.x[i][2]}];
    s = std::max(s, std::fabs(r.hi[i]));
  }
  double worst = 0;
  bench::Args at{};
  std::size_t bad = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double e = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / ulp_of(scale[{r.x[i][0], r.x[i][2]}]);
    if (!std::isfinite(e)) {
      ++bad;
      continue;
    }
    if (e > worst) worst = e, at = r.x[i];
  }
  INFO(name, ": max ulp of the component's scale at that a ", worst, " at (a ", at[0], ", x ", at[1], ", j = ", at[2], ")");
  CHECK(bad == 0);
  CHECK(worst <= bound);
}

// the limits at c = 0 and c = infinity, evaluated in a kernel
struct Lim {
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double c, double jd) const {
    return p ? laplace::math::log_gamma_p_ratio_mixed_d4(a, c).v[static_cast<int>(jd)]
             : laplace::math::log_gamma_q_ratio_mixed_d4(a, c).v[static_cast<int>(jd)];
  }
};

}  // namespace

TEST_CASE("partials of log P(a, a c) at fixed ratio against mpmath") {
  check_ratio(1, true, 2250);
  check_ratio(2, true, 3200);
  check_ratio(3, true, 3150);
  check_ratio(4, true, 2650);
  // gcc 2050, CUDA 4193 (a = 39.9, c = 0.9967, j = 5: below Temme's range, the quadrature
  // near c = 1, the loss left open in docs/numerics.md); 1.5 times the CUDA maximum
  check_ratio(5, true, 6300);
  check_ratio(6, true, 3960);
}

TEST_CASE("partials of log Q(a, a c) at fixed ratio against mpmath") {
  check_ratio(1, false, 2600);
  check_ratio(2, false, 2150);
  check_ratio(3, false, 1850);
  check_ratio(4, false, 1800);
  check_ratio(5, false, 2560);
  check_ratio(6, false, 2840);
}

TEST_CASE("mixed partials of log P and log Q at integer a near the mean against mpmath") {
  const double bp[6] = {7.1, 19.0, 170.0, 700.0, 2310.0, 6200.0};
  const double bq[6] = {6.0, 21.5, 109.0, 480.0, 1080.0, 2180.0};
  for (int k = 1; k <= 6; ++k) check_poisson(k, true, bp[k - 1]);
  for (int k = 1; k <= 6; ++k) check_poisson(k, false, bq[k - 1]);
}

TEST_CASE("partials at fixed ratio at c = 0 and c = infinity") {
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<bench::Args> args;
  for (int j = 0; j <= 4; ++j) args.push_back({2.5, 0.0, double(j)});
  for (int j = 0; j <= 4; ++j) args.push_back({2.5, inf, double(j)});
  const auto yp = bench::eval_kernel(args, Lim{true});
  const auto yq = bench::eval_kernel(args, Lim{false});
  // P at c = 0: d^4/da^4 (a log a - log Gamma(a + 1)) = 2/a^3 - psi'''(a + 1)
  CHECK(std::isfinite(yp[0]));
  CHECK(std::isfinite(yp[1]));
  CHECK(std::isfinite(yp[2]));
  CHECK(yp[3] == inf);
  CHECK(yp[4] == -inf);
  for (int j = 0; j <= 4; ++j) CHECK(yp[5 + j] == 0.0);
  CHECK(yq[0] == 0.0);
  CHECK(std::isfinite(yq[5]));
  for (int j = 1; j <= 4; ++j) CHECK(yq[5 + j] == 0.0);
}
