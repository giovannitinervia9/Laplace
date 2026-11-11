// Faa di Bruno's formula and the partial Bell polynomials to order 7
// (include/laplace/math/faa_di_bruno.hpp): the polynomials against Stirling
// numbers and the closed form B_(n,k)(1, 2, 3, ...) = C(n, k) k^(n-k); the
// composition, in a Kokkos kernel, against the mpmath references (exact on the
// double inputs). The error is measured in ulps of S = sum_k |f_k|
// B_(n,k)(|g|), the sum of the absolute values of the terms: where they cancel
// the result has no relative accuracy to offer (log(exp(x)) = x has zero
// derivatives from order two on), and the bound is the one measured on
// 2026-11-11 on both builds with a margin of about half again.
#include <doctest/doctest.h>

#include "faa_common.hpp"

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

TEST_CASE("partial Bell polynomials: Stirling numbers and C(n, k) k^(n-k)") {
  // Stirling numbers of the second kind S(n, k), n = 1..7
  const double S2[8][8] = {{0},
                           {0, 1},
                           {0, 1, 1},
                           {0, 1, 3, 1},
                           {0, 1, 7, 6, 1},
                           {0, 1, 15, 25, 10, 1},
                           {0, 1, 31, 90, 65, 15, 1},
                           {0, 1, 63, 301, 350, 140, 21, 1}};
  const double ones[8] = {0, 1, 1, 1, 1, 1, 1, 1}, nat[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  for (int n = 1; n <= 7; ++n)
    for (int k = 1; k <= n; ++k) {
      INFO("n = ", n, ", k = ", k);
      CHECK(laplace::math::bell_partial(n, k, ones) == S2[n][k]);
      double c = 1;  // C(n, k)
      for (int i = 0; i < k; ++i) c = c * (n - i) / (i + 1);
      CHECK(laplace::math::bell_partial(n, k, nat) == c * std::pow(double(k), n - k));
    }
  CHECK(laplace::math::bell_partial(3, 4, ones) == 0.0);
}

TEST_CASE("Faa di Bruno's formula against mpmath") {
  const auto r = faa::read(std::string(LAPLACE_REFERENCE_DIR) + "/faa_di_bruno.csv");
  const double bound[8] = {0, 1.0, 3.0, 3.5, 4.5, 4.5, 6.0, 7.5};
  for (int n = 1; n <= 7; ++n) {
    const auto e = faa::errors(r, faa::eval(r, n), n);
    INFO("compose_d", n, ": max ", e.ulpS, " ulp of the sum of |terms| (row ", e.worst, ")");
    CHECK(e.bad == 0);
    CHECK(e.ulpS <= bound[n]);
  }
}
