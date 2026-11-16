// The polygamma remainder psi^(n)(x) - mu_n(x)
// (own::polygamma_monomial_remainder of include/laplace/math/polygamma.hpp) against the
// mpmath references, and against the cancellation it exists to remove.
//
// mu_0(x) = log x and mu_n(x) = (-1)^(n+1) (n - 1)! x^-n are the leading terms
// of the asymptotic expansion, so the remainder is of relative size 1/(2x)
// against psi^(n) itself. The references come from
// tools/reference/psi_remainder.py, which performs that subtraction at a
// precision where it costs nothing, while the library never performs it at all:
// it drops the leading term of the expansion. The two routes share no
// arithmetic.
//
// Every kernel is in a named function with external linkage: an extended device
// lambda inside a doctest TEST_CASE shares its mangled tag with the same
// counter in another translation unit (CLAUDE.md, 2026-11-13).
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/polygamma.hpp>
#include <laplace/math/special.hpp>

#include <cmath>
#include <string>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;

struct Remainder {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return lm::own::polygamma_monomial_remainder(n, x); }
};

// mu_n(x), the leading term that the remainder leaves out.
double monomial(int n, double x) {
  if (n == 0) return std::log(x);
  double f = 1.0;
  for (int i = 2; i <= n - 1; ++i) f *= double(i);  // (n - 1)!
  double p = 1.0;
  for (int i = 0; i < n; ++i) p /= x;
  return ((n % 2) ? 1.0 : -1.0) * f * p;
}

void check_ref(const std::string& name, auto f, double max_ulp) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  const auto a = bench::accuracy(r, y);
  INFO(name, ": max ", a.max_ulp, " ulp at x = ", a.worst[0]);
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp <= max_ulp);
}

}  // namespace

TEST_CASE("polygamma remainder: every order against mpmath") {
  // Measured on gcc over the reference grid, x from 1e-3 to 1e12. Order two
  // reaches 14.8 ulp at x = 13, which is just below the recurrence threshold
  // X0 = 15: there the shift is by one, so the two monomials z^-n and x^-n are
  // comparable and their difference costs the digit the header predicts. The
  // first guess of 12 was below that and nothing else exceeded its own entry.
  // The CUDA build passes the same bounds (measured 2026-11-17).
  const double bound[7] = {12.0, 12.0, 22.0, 12.0, 16.0, 20.0, 24.0};
  for (int n = 0; n <= 6; ++n)
    check_ref("psi_remainder_d" + std::to_string(n), Remainder{n}, bound[n]);
}

TEST_CASE("polygamma remainder: the defining identity where it can be read") {
  // r_n(x) + mu_n(x) must be psi^(n)(x). This is only testable where the
  // remainder is not small against psi^(n), that is for x of order one, since
  // for large x the sum reconstructs a quantity whose last digits the
  // remainder no longer carries. That is the whole reason the function exists,
  // so the identity is a check of the small-x branch and not of the expansion.
  for (double x : {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}) {
    for (int n = 0; n <= 6; ++n) {
      const double r = lm::own::polygamma_monomial_remainder(n, x);
      const double p = lm::polygamma(n, x);
      const double m = monomial(n, x);
      INFO("n = ", n, ", x = ", x, ": remainder ", r, " + monomial ", m, " against psi ", p);
      CHECK(std::fabs(r + m - p) <= 1e-13 * std::fabs(p));
    }
  }
}

TEST_CASE("polygamma remainder: it removes the cancellation it was written for") {
  // The measurement that justifies the function, and the one that would notice
  // if it ever stopped working. The beta distribution in the mean mu and the
  // precision phi has
  //     I_phi,phi = mu^2 psi_1(alpha) + nu^2 psi_1(beta) - psi_1(phi),
  // with alpha = mu phi, beta = nu phi and nu = 1 - mu, whose leading terms
  // cancel exactly because mu^2/alpha + nu^2/beta = 1/phi. Written straight it
  // loses about log10(phi) digits; written with the monomials summed
  // algebraically (which is to say, not summed at all, since they cancel) and
  // only the remainders carried, it does not.
  //
  // The monomial of psi_1 at x is 1/x, so
  //     straight:  mu^2 psi_1(a) + nu^2 psi_1(b) - psi_1(s)
  //     by parts:  mu^2 r_1(a) + nu^2 r_1(b) - r_1(s)
  //                + [mu^2/a + nu^2/b - 1/s], the bracket being exactly 0.
  const double mu = 0.3, nu = 1.0 - mu;
  for (double phi : {1e2, 1e4, 1e6, 1e8}) {
    const double a = mu * phi, b = nu * phi, s = phi;
    const double straight = mu * mu * lm::polygamma(1, a) + nu * nu * lm::polygamma(1, b) - lm::polygamma(1, s);
    const double by_parts = mu * mu * lm::own::polygamma_monomial_remainder(1, a) +
                            nu * nu * lm::own::polygamma_monomial_remainder(1, b) -
                            lm::own::polygamma_monomial_remainder(1, s);
    // The true value is of size 1/phi^2 while the terms are of size 1/phi, so
    // the straight form keeps about eps*phi of relative error and the
    // by-parts form about eps. Both must agree to the straight form's own
    // accuracy, and the by-parts form must be the better of the two.
    INFO("phi = ", phi, ": straight ", straight, ", by parts ", by_parts);
    CHECK(by_parts > 0.0);  // I_phi,phi is positive
    CHECK(std::fabs(straight - by_parts) <= 4.0 * 2.3e-16 * phi * std::fabs(by_parts) + 1e-300);
    // and the by-parts form agrees with the leading behaviour of the true
    // value. With psi_1(x) = 1/x + 1/(2x^2) + 1/(6x^3) + O(x^-5),
    //   mu^2 psi_1(a) = mu/phi + 1/(2 phi^2) + 1/(6 mu phi^3) + ...
    //   nu^2 psi_1(b) = nu/phi + 1/(2 phi^2) + 1/(6 nu phi^3) + ...
    //      -psi_1(s)  =  -1/phi - 1/(2 phi^2) - 1/(6 phi^3)   + ...
    // so the 1/phi terms cancel through mu + nu = 1 and what is left is
    //   1/(2 phi^2) + (1/mu + 1/nu - 1)/(6 phi^3) + O(phi^-4).
    // An earlier version of this check wrote the whole leading term as
    // (1/mu + 1/nu - 1)/(2 phi^2), which conflates the two orders: that factor
    // belongs to the phi^-3 term over 6, not to the leading one over 2. It was
    // wrong by 3.7153 at phi = 100 (mu = 0.3), and 1 - 1/3.7153 = 0.731 is
    // exactly what this assertion then reported. Checked against mpmath at 30
    // digits: the corrected form is within 2.6e-6 of the true value at
    // phi = 100 and within 1e-9 at phi = 1e4.
    const double lead = 1.0 / (2.0 * phi * phi) + (1.0 / mu + 1.0 / nu - 1.0) / (6.0 * phi * phi * phi);
    CHECK(std::fabs(by_parts / lead - 1.0) <= 60.0 / (phi * phi));
  }
}

TEST_CASE("polygamma remainder: arguments outside the domain") {
  for (int n = 0; n <= 6; ++n) {
    CHECK(std::isnan(lm::own::polygamma_monomial_remainder(n, 0.0)));
    CHECK(std::isnan(lm::own::polygamma_monomial_remainder(n, -1.0)));
  }
  CHECK(std::isnan(lm::own::polygamma_monomial_remainder(-1, 1.0)));
  CHECK(std::isnan(lm::own::polygamma_monomial_remainder(7, 1.0)));
  // the remainder is negative for n = 0 (psi(x) < log x) and alternates with n
  // for x large, following (-1)^(n+1) n!/(2 x^(n+1))
  for (double x : {20.0, 1e3, 1e6}) {
    CHECK(lm::own::polygamma_monomial_remainder(0, x) < 0.0);
    CHECK(lm::own::polygamma_monomial_remainder(1, x) > 0.0);
    CHECK(lm::own::polygamma_monomial_remainder(2, x) < 0.0);
  }
}
