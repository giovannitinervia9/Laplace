// The von Mises block of Bessel functions (include/laplace/math/bessel_i.hpp)
// against the mpmath references, evaluated in a Kokkos kernel on the default
// execution space: log I0 and its scaled form, the ratio A = I1/I0 and its
// derivatives to order six, the inverse K of A and its derivatives to order
// six, and the ratios I_j/I_0 of the von Mises cdf series. Bounds: the errors
// measured on 2026-11-10 on the gcc/OpenMP and CUDA
// builds (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/bessel_i.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;

struct LogI0 {
  KOKKOS_INLINE_FUNCTION double operator()(double k) const { return lm::log_bessel_i0(k); }
};
struct LogI0Scaled {
  KOKKOS_INLINE_FUNCTION double operator()(double k) const { return lm::log_bessel_i0_scaled(k); }
};
struct Ratio {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double k) const {
    switch (n) {
      case 0: return lm::bessel_i_ratio(k);
      case 1: return lm::bessel_i_ratio_d1(k);
      case 2: return lm::bessel_i_ratio_d2(k);
      case 3: return lm::bessel_i_ratio_d3(k);
      case 4: return lm::bessel_i_ratio_d4(k);
      case 5: return lm::bessel_i_ratio_d5(k);
      default: return lm::bessel_i_ratio_d6(k);
    }
  }
};
struct Inverse {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (n) {
      case 0: return lm::bessel_i_ratio_inv(x);
      case 1: return lm::bessel_i_ratio_inv_d1(x);
      case 2: return lm::bessel_i_ratio_inv_d2(x);
      case 3: return lm::bessel_i_ratio_inv_d3(x);
      case 4: return lm::bessel_i_ratio_inv_d4(x);
      case 5: return lm::bessel_i_ratio_inv_d5(x);
      default: return lm::bessel_i_ratio_inv_d6(x);
    }
  }
};

// R_j = I_j/I_0 as the von Mises cdf series uses them: all m ratios at once,
// m = max(20, ceil(8.5 sqrt k) + 10), and the one asked for returned
struct Ratios {
  KOKKOS_INLINE_FUNCTION double operator()(double k, double jd) const {
    const int j = int(jd);
    int m = int(Kokkos::ceil(8.5 * Kokkos::sqrt(k))) + 10;
    if (m < 20) m = 20;
    if (j > m) m = j;
    double out[512];
    if (m > 512) return k * NAN;
    lm::bessel_i_ratios(k, m, out);
    return out[j - 1];
  }
};

void check(const std::string& name, auto f, double max_ulp) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  const auto a = bench::accuracy(r, y);
  INFO(name, ": max ", a.max_ulp, " ulp at ", a.worst[0]);
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp <= max_ulp);
}

}  // namespace

TEST_CASE("log I0 and log I0 - k against mpmath") {
  check("log_bessel_i0", LogI0{}, 3.5);
  check("log_bessel_i0_scaled", LogI0Scaled{}, 4.5);
}

TEST_CASE("A = I1/I0 and its derivatives against mpmath") {
  const double bound[7] = {5.0, 3.0, 3.5, 3.0, 3.5, 3.0, 3.5};
  check("bessel_i_ratio", Ratio{0}, bound[0]);
  for (int n = 1; n <= 6; ++n) check("bessel_i_ratio_d" + std::to_string(n), Ratio{n}, bound[n]);
}

TEST_CASE("the inverse of A and its derivatives against mpmath") {
  const double bound[7] = {3.0, 3.0, 3.5, 3.5, 3.0, 4.0, 4.5};
  check("bessel_i_ratio_inv", Inverse{0}, bound[0]);
  for (int n = 1; n <= 6; ++n) check("bessel_i_ratio_inv_d" + std::to_string(n), Inverse{n}, bound[n]);
}

TEST_CASE("the ratios I_j/I_0 of the von Mises cdf series against mpmath") {
  const auto r = bench::read_ref(refdir + "/bessel_i_ratios.csv");
  const auto y = bench::eval_kernel(r.x, Ratios{});
  const auto a = bench::accuracy(r, y);
  INFO("bessel_i_ratios: max ", a.max_ulp, " ulp, ", a.max_ulp1, " ulp1 at k = ", a.worst[0], ", j = ", a.worst[1]);
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp <= 23.0);
  CHECK(a.max_ulp1 <= 3.0);
}

TEST_CASE("special arguments of the von Mises Bessel functions") {
  CHECK(lm::bessel_i_ratio(0.0) == 0.0);
  CHECK(lm::bessel_i_ratio_inv(0.0) == 0.0);
  CHECK(std::isinf(lm::bessel_i_ratio_inv(1.0)));
  CHECK(lm::log_bessel_i0(0.0) == 0.0);
  CHECK(std::isnan(lm::bessel_i_ratio(-1.0)));
  CHECK(std::isnan(lm::bessel_i_ratio_inv(1.5)));
  CHECK(std::isnan(lm::log_bessel_i0(-1.0)));
  double out[3];
  lm::bessel_i_ratios(0.0, 3, out);
  CHECK((out[0] == 0.0 && out[2] == 0.0));
  lm::bessel_i_ratios(1e-200, 3, out);  // I_1/I_0 = k/2 at tiny k; the rest underflow
  CHECK(out[0] == 5e-201);
  CHECK(out[2] == 0.0);
  lm::bessel_i_ratios(2.0, 3, out);  // the first ratio is A
  CHECK(std::fabs(out[0] - lm::bessel_i_ratio(2.0)) <= 4e-16 * out[0]);
  // A(K(rho)) = rho to within the conditioning of K
  for (double rho : {1e-10, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-12}) {
    const double k = lm::bessel_i_ratio_inv(rho);
    CHECK(std::fabs(lm::bessel_i_ratio(k) - rho) <= 4e-16 * rho);
  }
}
