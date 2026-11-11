// The special functions of laplace::math against the mpmath references,
// evaluated in a Kokkos kernel on the default execution space. The bounds are
// the errors measured on 2026-11-08 (docs/numerics.md) with a margin of about
// half again: a regression shows up as a failure, a platform that is slightly
// worse does not.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/special.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;

struct Lgamma {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::lgamma(x); }
};
struct Polygamma {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::polygamma(n, x); }
};

struct GammaP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::gamma_p(a, x); }
};
struct GammaQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::gamma_q(a, x); }
};

struct LogGammaP {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::log_gamma_p(a, x); }
};
struct LogGammaQ {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x) const { return laplace::math::log_gamma_q(a, x); }
};

struct Ibeta {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const { return laplace::math::ibeta(a, b, x); }
};
struct Ibetac {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const { return laplace::math::ibetac(a, b, x); }
};

// max relative error (ulp) and max absolute error in ulps of max(|f|, 1)
void check(const char* name, auto f, double max_ulp, double max_ulp1, bool relative_everywhere,
           double max_ulpL = 1e300) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  const auto a = bench::accuracy(r, y);
  INFO(std::string(name), ": max ", a.max_ulp, " ulp, max ", a.max_ulp1, " ulp1, ", a.max_ulpL, " ulpL, worst x ", a.worst[0]);
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp1 <= max_ulp1);
  if (relative_everywhere) CHECK(a.max_ulp <= max_ulp);
  CHECK(a.max_ulpL <= max_ulpL);
}

}  // namespace

TEST_CASE("lgamma, digamma, trigamma and polygamma against mpmath") {
  check("lgamma", Lgamma{}, 4.0, 4.0, true);
  // digamma: relative error unbounded at its root, only the absolute error is bounded
  check("digamma", Polygamma{0}, 0.0, 3.5, false);
  check("trigamma", Polygamma{1}, 3.0, 3.0, true);
  check("polygamma2", Polygamma{2}, 5.0, 5.0, true);
  check("polygamma3", Polygamma{3}, 8.0, 8.0, true);
  check("polygamma4", Polygamma{4}, 9.5, 9.5, true);
  check("polygamma5", Polygamma{5}, 12.0, 12.0, true);
  check("polygamma6", Polygamma{6}, 14.0, 14.0, true);
}

// The tails of P and Q carry the error of exp(-a phi), about |log P| ulps times
// a small factor: bounded through ulpL = ulp / max(1, |log P|).
TEST_CASE("incomplete gamma against mpmath") {
  check("gamma_p", GammaP{}, 0.0, 12.0, false, 45.0);
  check("gamma_q", GammaQ{}, 0.0, 12.0, false, 40.0);
}

// Large shapes, a from 1e2 to 1e15 with x/a - 1 from 2^-40 to 1 in magnitude,
// where Temme's uniform expansion is used for |eta| <= 0.5: P and Q in ulps of
// max(|f|, 1) and in ulpL (the small tail carries about a phi = |log tail|
// ulps from its argument), the logarithms in ulps of max(|f|, 1).
TEST_CASE("incomplete gamma for large shapes against mpmath") {
  check("gamma_p_large", GammaP{}, 0.0, 1.5, false, 4.5);
  check("gamma_q_large", GammaQ{}, 0.0, 1.5, false, 5.5);
  check("log_gamma_p_large", LogGammaP{}, 0.0, 9.0, false);
  check("log_gamma_q_large", LogGammaQ{}, 0.0, 10.0, false);
}

// Measured 2026-11-09 with DiDonato and Morris' expansions
// (detail/ibeta_expansions.hpp): ibeta ulp1 2.82 (CPU) and 3.80 (CUDA), ulpL
// 12.7 and 12.9; ibetac ulp1 2.82 and 3.80, ulpL 16.9 and 12.2. The first own
// version (continued fraction only) reached ulp1 216-331 and ulpL 1311-1622.
TEST_CASE("incomplete beta against mpmath") {
  check("ibeta", Ibeta{}, 0.0, 5.7, false, 19.5);
  check("ibetac", Ibetac{}, 0.0, 5.7, false, 25.5);
}
