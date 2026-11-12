// The quadrature rules (include/laplace/math/quadrature.hpp), in a Kokkos
// kernel on the default execution space: each rule on the integrals of
// tools/reference/quadrature.py it is meant for, with the error measured on
// 2026-11-11 on both builds plus a margin of about half again, and the fixed
// Gauss rules on the polynomials they integrate exactly.
#include <doctest/doctest.h>

#include "quad_common.hpp"

#include <cmath>
#include <limits>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

// max ulp of rule m on integrand id, and the number of rows that did not converge
void check(const qbench::Ref& r, const qbench::Out& o, int m, int id, double bound) {
  double worst = 0;
  int nconv = 0, cnt = 0;
  for (std::size_t i = 0; i < r.id.size(); ++i) {
    if (r.id[i] != id) continue;
    ++cnt;
    if (!o.conv[i]) ++nconv;
    const double e = std::fabs((o.value[i] - r.hi[i]) - r.lo[i]) / bench::ulp_of(r.hi[i]);
    worst = std::max(worst, std::isfinite(e) ? e : 1e300);
  }
  INFO(std::string(qbench::method_name(m)), " on integrand ", id, ": max ", worst, " ulp, ", nconv, " of ", cnt, " not converged");
  CHECK(cnt > 0);
  CHECK(nconv == 0);
  CHECK(worst <= bound);
}

}  // namespace

TEST_CASE("quadrature rules against mpmath, each on the integrals it is meant for") {
  const auto r = qbench::read(std::string(LAPLACE_REFERENCE_DIR) + "/quadrature.csv");
  const double stop = laplace::math::quad::de_quad_options{}.stop;
  {
    const auto o = qbench::eval(r, qbench::TS, stop);
    check(r, o, qbench::TS, 0, 10.0);  // beta, singular ends, with the distance to the end
    check(r, o, qbench::TS, 5, 10.0);  // x^p log(x)^2
  }
  {
    const auto o = qbench::eval(r, qbench::ES, stop);
    check(r, o, qbench::ES, 1, 100.0);  // gamma (68 ulp where the peak is far from the end)
    check(r, o, qbench::ES, 3, 8.0);    // Student t tail
  }
  {
    const auto o = qbench::eval(r, qbench::SS, stop);
    check(r, o, qbench::SS, 6, 11.0);  // skew normal information entry
    check(r, o, qbench::SS, 7, 27.0);  // pseudo-Huber normalizer
    check(r, o, qbench::SS, 8, 10.0);  // sech^p
  }
  {
    const auto o = qbench::eval(r, qbench::PAN, stop);
    check(r, o, qbench::PAN, 1, 32.0);
    check(r, o, qbench::PAN, 3, 6.0);
    check(r, o, qbench::PAN, 4, 7.0);  // Laplace density truncated, kink at 0
    check(r, o, qbench::PAN, 5, 6.0);
    check(r, o, qbench::PAN, 6, 10.0);
    check(r, o, qbench::PAN, 7, 5.0);
    check(r, o, qbench::PAN, 8, 6.0);
  }
  {
    const auto o = qbench::eval(r, qbench::GK, stop);
    check(r, o, qbench::GK, 4, 110.0);
  }
}

namespace {
struct Mono {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    double p = 1.0;
    for (int i = 0; i < k; ++i) p *= x;
    return p;
  }
};
struct GaussMono {  // x^k e^(-x^2)
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Mono{k}(x) * Kokkos::exp(-x * x); }
};
struct ExpMono {  // x^k e^-x
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return Mono{k}(x) * Kokkos::exp(-x); }
};
}  // namespace

TEST_CASE("fixed Gauss rules integrate their polynomials") {
  namespace quad = laplace::math::quad;
  // int_0^1 x^k = 1/(k+1); Gauss-Legendre with n nodes is exact to degree 2n - 1
  for (int k : {0, 1, 7, 31}) {
    INFO("k = ", k);
    CHECK(quad::gauss_legendre_quad<16>(Mono{k}, 0.0, 1.0) == doctest::Approx(1.0 / (k + 1)).epsilon(4e-16));
    CHECK(quad::gauss_legendre_quad<64>(Mono{k}, 0.0, 1.0) == doctest::Approx(1.0 / (k + 1)).epsilon(4e-16));
  }
  // int_R x^(2j) e^(-x^2) = Gamma(j + 1/2); int_0^inf x^k e^-x = k!
  const double g[] = {1.7724538509055160273, 0.88622692545275801365, 1.3293403881791370205, 3.3233509704478425512};
  for (int j = 0; j < 4; ++j) {
    INFO("j = ", j);
    CHECK(quad::gauss_hermite32_quad(GaussMono{2 * j}) == doctest::Approx(g[j]).epsilon(1e-14));
  }
  double fact = 1.0;
  for (int k = 0; k < 10; ++k) {
    if (k > 0) fact *= k;
    INFO("k = ", k);
    CHECK(quad::gauss_laguerre32_quad(ExpMono{k}) == doctest::Approx(fact).epsilon(1e-14));
  }
}

TEST_CASE("quadrature rules: reversed, empty and invalid arguments") {
  namespace quad = laplace::math::quad;
  const double nan = std::nan(""), inf = std::numeric_limits<double>::infinity();
  auto sq = [](double x) { return x * x; };            // int_0^1 = 1/3
  auto g = [](double x) { return std::exp(-x * x); };  // int over R = sqrt(pi)
  auto e = [](double x) { return std::exp(-x); };      // int_0^inf = 1
  // int_a^b = -int_b^a
  const auto ts = quad::tanh_sinh_quad(sq, 0.0, 1.0), tr = quad::tanh_sinh_quad(sq, 1.0, 0.0);
  CHECK(tr.converged);
  CHECK(tr.value == -ts.value);
  CHECK(tr.error == ts.error);
  CHECK(quad::gauss_kronrod_quad(sq, 1.0, 0.0).value == doctest::Approx(-1.0 / 3.0).epsilon(1e-15));
  CHECK(quad::gauss_legendre_quad<16>(sq, 1.0, 0.0) == doctest::Approx(-1.0 / 3.0).epsilon(4e-16));
  const auto pf = quad::panels_quad(sq, 0.0, 1.0, 0.0, 0.25), pr = quad::panels_quad(sq, 1.0, 0.0, 0.0, 0.25);
  CHECK(pr.converged);
  CHECK(pr.value == -pf.value);
  // empty interval: exactly 0, error 0, converged
  for (const auto& r : {quad::tanh_sinh_quad(sq, 0.5, 0.5), quad::gauss_kronrod_quad(sq, 0.5, 0.5), quad::panels_quad(sq, 0.5, 0.5, 0.0, 1.0)}) {
    CHECK(r.value == 0.0);
    CHECK(r.error == 0.0);
    CHECK(r.converged);
  }
  // invalid arguments: NaN, not converged
  for (const auto& r : {quad::tanh_sinh_quad(sq, nan, 1.0), quad::tanh_sinh_quad(sq, 0.0, inf), quad::gauss_kronrod_quad(sq, 0.0, nan),
                        quad::exp_sinh_quad(e, 0.0, 0.0), quad::exp_sinh_quad(e, 0.0, -1.0), quad::exp_sinh_quad(e, nan, 1.0),
                        quad::sinh_sinh_quad(g, 0.0, -1.0), quad::sinh_sinh_quad(g, nan, 1.0), quad::sinh_sinh_quad(g, 0.0, inf),
                        quad::panels_quad(sq, nan, 1.0, 0.0, 1.0), quad::panels_quad(sq, 0.0, 1.0, 0.0, 0.0)}) {
    CHECK(std::isnan(r.value));
    CHECK_FALSE(r.converged);
  }
  CHECK(std::isnan(quad::gauss_hermite32_quad(sq, 0.0, -1.0)));
  CHECK(std::isnan(quad::gauss_laguerre32_quad(sq, 0.0, 0.0)));
  CHECK(std::isnan(quad::gauss_hermite32_quad(sq, nan, 1.0)));
  // valid scales still give the integral
  CHECK(quad::sinh_sinh_quad(g, 0.0, 1.0).value == doctest::Approx(std::sqrt(3.14159265358979323846)).epsilon(1e-14));
  CHECK(quad::exp_sinh_quad(e, 0.0, 1.0).value == doctest::Approx(1.0).epsilon(1e-14));
}
