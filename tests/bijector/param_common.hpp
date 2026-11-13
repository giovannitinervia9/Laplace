// Shared check of a scalar bijector with parameters against its mpmath
// references (tests/data/reference/bijector_<map>.csv, written by
// tools/reference/bijector_<map>.py through bijector_param_common.py),
// evaluated in a Kokkos kernel on the default execution space.
//
// Every row is a quantity (which: 0 forward, 1 inverse, 2 forward log-Jacobian,
// 3 inverse log-Jacobian, 4 and 5 the distances to the upper and lower bound)
// and a derivative d^k / (d arg^n0 d p1^n1 d p2^n2), k = n0 + n1 + n2 <= 6, at
// (x, p1, p2). Order 0 is the value, pure derivatives (n1 = n2 = 0) go through
// forward_dK / inverse_dK / ..._log_jacobian_dK, mixed ones through the
// _partials_dK functions; for the pure rows the component c[0] of the partials
// is checked as well. Error in ulpC: ulps of
//   |f| + |x df/dx| + |p1 df/dp1| + |p2 df/dp2|,
// the change that one ulp of relative perturbation of every input would cause
// (the neighbouring derivatives are read from the same file; the distances use
// the derivatives of h). Set LAPLACE_PRINT_ERRORS=1 to print the table.
#pragma once

#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/bijector/scalar.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace bijector_param_test {

namespace lb = laplace::bijector;
using map_t = lb::scalar_bijector_map;

// the bijector of a map with its parameters set directly (no normalization)
KOKKOS_INLINE_FUNCTION lb::scalar_bijector make(map_t m, double p1, double p2) {
  lb::scalar_bijector b{m};
  switch (m) {
    case map_t::affine: b.location = p1, b.scale = p2; break;
    case map_t::power:
    case map_t::box_cox:
    case map_t::yeo_johnson: b.exponent = p1; break;
    case map_t::softplus: b.sharpness = p1; break;
    case map_t::bounded: b.lower = p1, b.upper = p2; break;
    case map_t::lower_bounded: b.lower = p1; break;
    case map_t::upper_bounded: b.upper = p1; break;
    default: break;
  }
  return b;
}

// Checks the map m against bijector_<name>.csv; bound[w][k] is the largest
// ulpC accepted for quantity w at total order k (< 0: no rows expected).
// Defined in param_common.cpp.
void check_map(const char* name, map_t m, const double (&bound)[6][7]);

// a(i, .) = (x, p1, p2, which, n0, n1, n2) of row i; y(i, .) its two results
using Rows = Kokkos::View<double* [7], Kokkos::LayoutRight, bench::Exec::memory_space>;
using Out = Kokkos::View<double* [2], Kokkos::LayoutRight, bench::Exec::memory_space>;

// Evaluates the rows of total order K (the kernel of check_map); defined in
// param_order.cpp, one translation unit per K.
template <int K>
void eval_order(map_t m, Rows a, Out y);

}  // namespace bijector_param_test
