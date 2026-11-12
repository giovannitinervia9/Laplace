// Derivatives of log P(a, x) and log Q(a, x) with respect to a, orders 1 to 6
// (include/laplace/math/incomplete_gamma_deriv.hpp), against the mpmath
// references, evaluated in a Kokkos kernel on the default execution space.
// Bounds: the errors measured on 2026-11-09 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again. Measured, CPU / CUDA,
// with Temme's expansion, the fraction in Taylor form and the lower series for
// log P at a < 2, x < 3/2 (late evening of 2026-11-09):
// P orders 1-5: 4.0/4.3, 4.4/4.4, 8.8/10.8, 23.5/23.0, 197/171;
// Q orders 1-5: 4.8/4.8, 5.5/7.3, 13.9/13.9, 22.1/23.0, 62.9/67.9
// (the bounds of P orders 2-4 were 22, 58 and 85 before; order 5 was 420
// before the evening).
// Order 6 (2026-11-12, gcc only so far; 1900 points, the 1500 of orders 1-5
// plus 400 at the route boundaries): P 771 (the lower series at a = 0.0139,
// x = 1.045, where kappa_6 and psi^(5)(a + 1), both near 113, cancel to 1.7),
// Q 96.3 (the complement of the lower series at a = 2.04, x = 1.56).
//
// The figure per point is the smaller of ulpL = ulp / max(1, |log|f||) (in
// the far tails the complement log(1 - S) inherits the error of
// S = exp(log S), about |log S| ulps, which the logarithm of the result
// measures) and ulp1 = the error in ulps of max(|f|, 1) (where the
// derivatives of order 3 to 5 cross zero); the bound is on its maximum.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/incomplete_gamma_deriv.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;

struct Deriv {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const {
    namespace lm = laplace::math;  // the wrappers
    if (p) {
      switch (k) {
        case 1: return lm::log_gamma_p_da1(a, x);
        case 2: return lm::log_gamma_p_da2(a, x);
        case 3: return lm::log_gamma_p_da3(a, x);
        case 4: return lm::log_gamma_p_da4(a, x);
        case 5: return lm::log_gamma_p_da5(a, x);
        default: return lm::log_gamma_p_da6(a, x);
      }
    }
    switch (k) {
      case 1: return lm::log_gamma_q_da1(a, x);
      case 2: return lm::log_gamma_q_da2(a, x);
      case 3: return lm::log_gamma_q_da3(a, x);
      case 4: return lm::log_gamma_q_da4(a, x);
      case 5: return lm::log_gamma_q_da5(a, x);
      default: return lm::log_gamma_q_da6(a, x);
    }
  }
};

void check(int k, bool p, double bound) {
  const std::string name = std::string(p ? "log_gamma_p_da" : "log_gamma_q_da") + std::to_string(k);
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, Deriv{k, p});
  double worst = 0;
  bench::Args at{};
  std::size_t bad = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double hi = r.hi[i];
    const double e = std::fabs(bench::error_in(y[i], hi, r.lo[i], std::max(std::fabs(hi), 2.2250738585072014e-308)));
    if (!std::isfinite(e)) {
      ++bad;
      continue;
    }
    const double eL = e / std::max(1.0, std::fabs(std::log(std::fabs(hi))));
    const double e1 = std::fabs(bench::error_in(y[i], hi, r.lo[i], std::max(std::fabs(hi), 1.0)));
    const double u = std::min(eL, e1);
    if (u > worst) worst = u, at = r.x[i];
  }
  INFO(name, ": max min(ulpL, ulp1) ", worst, " at (", at[0], ", ", at[1], ")");
  CHECK(bad == 0);
  CHECK(worst <= bound);
}

}  // namespace

TEST_CASE("derivatives of log P(a, x) in a against mpmath") {
  check(1, true, 6.0);
  check(2, true, 7.0);
  check(3, true, 16.0);
  check(4, true, 35.0);
  check(5, true, 245.0);
  check(6, true, 1160.0);
}

TEST_CASE("derivatives of log Q(a, x) in a against mpmath") {
  check(1, false, 7.5);
  check(2, false, 11.0);
  check(3, false, 21.0);
  check(4, false, 35.0);
  check(5, false, 102.0);
  check(6, false, 145.0);
}
