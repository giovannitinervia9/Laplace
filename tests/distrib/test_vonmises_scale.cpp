// The map from the von Mises circular standard deviation s to the concentration
// kappa, with its derivatives to order six
// (include/laplace/distrib/vonmises_scale.hpp), against the mpmath
// references and against its own three routes.
//
// The references come from tools/reference/vonmises_scale.py, whose derivatives
// are built by reverting the Taylor series of A = I_1/I_0 and composing it with
// the exact series of exp(-s^2/2); that route agrees with mpmath's own
// numerical differentiation of the map to 1e-79 relative, so the reference is
// settled by two independent routes before the library is compared with it.
//
// Every kernel lives in a named function with external linkage: an extended
// device lambda inside a doctest TEST_CASE shares its mangled tag with the same
// counter in another translation unit (CLAUDE.md, 2026-11-13).
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/distrib/vonmises_scale.hpp>
#include <laplace/math/bessel_i.hpp>

#include <cmath>
#include <string>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace ld = laplace::distrib;
namespace lm = laplace::math;

// The value and the six orders, one functor over the order so that the kernel
// calls the per-order functions, which are the ones the references check.
struct Kappa {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    switch (n) {
      case 0: return ld::vonmises_kappa_from_s(s);
      case 1: return ld::vonmises_kappa_from_s_dk<1>(s);
      case 2: return ld::vonmises_kappa_from_s_dk<2>(s);
      case 3: return ld::vonmises_kappa_from_s_dk<3>(s);
      case 4: return ld::vonmises_kappa_from_s_dk<4>(s);
      case 5: return ld::vonmises_kappa_from_s_dk<5>(s);
      default: return ld::vonmises_kappa_from_s_dk<6>(s);
    }
  }
};

// The joint routine, which must give exactly what the per-order ones give.
struct KappaUpto {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    double out[7];
    ld::vonmises_kappa_from_s_upto<6>(s, out);
    return out[n];
  }
};

void check_ref(const std::string& name, auto f, double max_ulp) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  const auto a = bench::accuracy(r, y);
  INFO(name, ": max ", a.max_ulp, " ulp at s = ", a.worst[0]);
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp <= max_ulp);
}

}  // namespace

// The errors measured on 2026-11-15 over the reference grid, s from 1e-8 to
// 4.5, worst of the gcc/OpenMP and CUDA builds, with a margin of about half
// again. The worst point of every order but the value is s = 0.2, the low end
// of the band where the inverse of the ratio and the Faa di Bruno composition
// do the work; the value's is s = 4.1, just below the switch to the series in
// rho.
const double kappa_bound[7] = {8.0, 10.0, 15.0, 20.0, 32.0, 40.0, 50.0};

TEST_CASE("vonmises map: kappa(s) and its six orders against mpmath") {
  const double* bound = kappa_bound;
  check_ref("vonmises_kappa", Kappa{0}, bound[0]);
  for (int n = 1; n <= 6; ++n) check_ref("vonmises_kappa_d" + std::to_string(n), Kappa{n}, bound[n]);
}

TEST_CASE("vonmises map: the joint routine agrees with the per-order ones") {
  const double* bound = kappa_bound;
  check_ref("vonmises_kappa", KappaUpto{0}, bound[0]);
  for (int n = 1; n <= 6; ++n)
    check_ref("vonmises_kappa_d" + std::to_string(n), KappaUpto{n}, bound[n]);
}

TEST_CASE("vonmises map: the complement entry point agrees with the inverse") {
  // The two entry points must agree wherever both receive an exact argument.
  // That means taking the complement q as the given quantity and q in
  // [0.5, 1], where 1 - q is exact by Sterbenz's lemma, so neither route
  // rounds its input. Outside that band one of the two inputs is rounded and a
  // difference is the conditioning of the rounded one, not a defect: at
  // rho = 1e-8, passing q = 1 - rho rounds rho itself to 1.1e-8 relative, and
  // the complement route then answers a slightly different question.
  for (double q : {0.5, 0.6, 0.75, 0.9, 0.99, 1.0 - 1e-15}) {
    const double a = lm::bessel_i_ratio_inv(1.0 - q);
    const double b = lm::bessel_i_ratio_inv_c(q);
    INFO("q = ", q, ": ", a, " against ", b);
    CHECK(std::fabs(a - b) <= 8e-16 * a);
  }
  // The defining identity, where it can be read at all. Checking it as
  // 1 - A(kappa) against q destroys the very quantity the entry point exists to
  // keep: for q below about 1e-8 the subtraction leaves nothing, so the
  // identity is only testable that way for q of order one, and the small-q end
  // is checked against the asymptote 1 - A = 1/(2 kappa) + 1/(8 kappa^2)
  // instead, whose own next term is the tolerance.
  for (double q : {1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-12}) {
    const double k = lm::bessel_i_ratio_inv_c(q);
    INFO("q = ", q, ": kappa = ", k, ", 1 - A(kappa) against q");
    CHECK(std::fabs((1.0 - lm::bessel_i_ratio(k)) / q - 1.0) <= 1e-12);
  }
  for (double q : {1e-15, 1e-10, 1e-8, 1e-6}) {
    const double k = lm::bessel_i_ratio_inv_c(q);
    const double asym = 0.5 / k * (1.0 + 0.25 / k);  // 1/(2k) + 1/(8k^2)
    INFO("q = ", q, ": kappa = ", k, ", asymptotic complement ", asym);
    // the asymptote's own next term is 1/(8 kappa^3), but below it lies the
    // floor of double arithmetic, so the tolerance is the larger of the two
    CHECK(std::fabs(asym / q - 1.0) <= 8e-16 + 4.0 / (k * k));
  }
  // and the complement route keeps kappa where the rho route cannot: at
  // q = 1e-17 the complement formed as 1 - rho would be rounded to zero
  CHECK(std::isinf(lm::bessel_i_ratio_inv(1.0 - 1e-17)));  // rho rounds to one
  const double k = lm::bessel_i_ratio_inv_c(1e-17);
  INFO("kappa at q = 1e-17: ", k);
  CHECK(std::isfinite(k));
  CHECK(std::fabs(k / 5e16 - 1.0) <= 1e-8);  // kappa ~ 1/(2q)
}

TEST_CASE("vonmises map: beyond s = 4.5 the conditioning of the map is the bound") {
  // There is no reference file above s = 4.5, and there should not be: kappa is
  // about 2 exp(-s^2/2), so d kappa / kappa = -s^2 (ds / s) and a relative
  // perturbation of s by one eps moves kappa by s^2 eps, whatever route
  // computes it. At s = 17.8 that is 178 ulp, which is what an earlier run of
  // this test measured and what a reference there would be comparing against.
  // So the check is against the leading asymptote with that tolerance, which is
  // the only honest statement the region admits.
  for (double s : {5.0, 8.0, 12.0, 17.782794, 20.0}) {
    const double kap = ld::vonmises_kappa_from_s(s);
    const double rho = std::exp(-0.5 * s * s);
    const double lead = 2.0 * rho + rho * rho * rho;  // the next term is 5 rho^5/6
    const double tol = 4.0 * s * s * 2.3e-16;
    INFO("s = ", s, ": kappa = ", kap, " against 2 rho + rho^3 = ", lead, ", tol ", tol);
    CHECK(std::fabs(kap / lead - 1.0) <= tol);
  }
  // and the derivatives keep their sign and their asymptotic size there. The
  // asymptote must carry the rho^3 term as the value's does: differentiating
  // 2 rho + rho^3 gives -s rho (2 + 3 rho^2), and dropping the second term
  // would leave a relative gap of 1.5 rho^2, which at s = 5 is 2e-11 and far
  // above the tolerance below.
  for (double s : {5.0, 8.0, 12.0}) {
    const double rho = std::exp(-0.5 * s * s);
    const double lead = -s * rho * (2.0 + 3.0 * rho * rho);
    const double d1 = ld::vonmises_kappa_from_s_dk<1>(s);
    INFO("s = ", s, ": kappa' = ", d1, " against -s rho (2 + 3 rho^2) = ", lead);
    CHECK(d1 < 0.0);
    CHECK(std::fabs(d1 / lead - 1.0) <= 4.0 * s * s * 2.3e-16);
  }
}

TEST_CASE("vonmises map: the three routes agree where they overlap") {
  // The series in s^2, the inverse from the complement and the series in rho
  // each hold beyond their switch point, so they must agree across it. A
  // disagreement here says a switch point is in the wrong place.
  namespace dv = ld::detail_vonmises;
  for (double s : {0.03, 0.04, 0.05, 0.06, 0.07}) {
    const double a = dv::value_small(s), b = dv::value_middle(s);
    INFO("at s = ", s, " small ", a, " middle ", b);
    CHECK(std::fabs(a / b - 1.0) <= 1e-14);
  }
  for (double s : {3.8, 3.9, 4.0, 4.1, 4.2}) {
    const double a = dv::value_large(s), b = dv::value_middle(s);
    INFO("at s = ", s, " large ", a, " middle ", b);
    CHECK(std::fabs(a / b - 1.0) <= 1e-14);
  }
}

TEST_CASE("vonmises map: the defining identity and the two asymptotes") {
  // A(kappa(s)) = exp(-s^2/2) to within the conditioning of the complement
  for (double s : {0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0}) {
    const double kap = ld::vonmises_kappa_from_s(s);
    const double rho = std::exp(-0.5 * s * s);
    INFO("s = ", s, ": A(kappa) = ", lm::bessel_i_ratio(kap), " against ", rho);
    CHECK(std::fabs(lm::bessel_i_ratio(kap) / rho - 1.0) <= 1e-13);
  }
  // kappa s^2 -> 1 as s -> 0, and kappa / (2 rho) -> 1 as s -> infinity
  for (double s : {1e-8, 1e-6, 1e-4}) {
    const double kap = ld::vonmises_kappa_from_s(s);
    CHECK(std::fabs(kap * s * s - 1.0) <= 1e-7);
  }
  for (double s : {6.0, 10.0, 20.0}) {
    const double kap = ld::vonmises_kappa_from_s(s);
    CHECK(std::fabs(kap / (2.0 * std::exp(-0.5 * s * s)) - 1.0) <= 1e-10);
  }
  // the first derivative is negative throughout: kappa falls as s grows
  for (double s : {1e-6, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0})
    CHECK(ld::vonmises_kappa_from_s_dk<1>(s) < 0.0);
  // kappa' = -2/s^3 as s -> 0
  for (double s : {1e-5, 1e-4, 1e-3}) {
    const double d1 = ld::vonmises_kappa_from_s_dk<1>(s);
    CHECK(std::fabs(d1 * s * s * s / -2.0 - 1.0) <= 1e-5);
  }
}

TEST_CASE("vonmises map: the information in (mu, s) from the map") {
  // I_ss = s^2 rho^2 / A'(kappa), which the plan derives from
  // drho/ds = -s rho = A'(kappa) kappa'(s); the identity is the one the
  // family will use, so check it here where the map is.
  for (double s : {0.01, 0.1, 0.5, 1.0, 2.0, 3.0}) {
    const double kap = ld::vonmises_kappa_from_s(s);
    const double rho = std::exp(-0.5 * s * s);
    const double ap = lm::bessel_i_ratio_d1(kap);
    const double d1 = ld::vonmises_kappa_from_s_dk<1>(s);
    INFO("s = ", s, ": kappa' = ", d1, " against -s rho / A' = ", -s * rho / ap);
    CHECK(std::fabs(d1 * ap / (-s * rho) - 1.0) <= 1e-12);
    const double i_ss_a = d1 * d1 * ap, i_ss_b = s * s * rho * rho / ap;
    CHECK(std::fabs(i_ss_a / i_ss_b - 1.0) <= 1e-12);
  }
  // as s -> 0 the information in s is that of a normal standard deviation
  for (double s : {1e-3, 1e-2}) {
    const double kap = ld::vonmises_kappa_from_s(s);
    const double rho = std::exp(-0.5 * s * s);
    const double i_ss = s * s * rho * rho / lm::bessel_i_ratio_d1(kap);
    CHECK(std::fabs(i_ss * s * s / 2.0 - 1.0) <= 1e-3);
  }
}

TEST_CASE("vonmises map: arguments outside the domain") {
  CHECK(std::isinf(ld::vonmises_kappa_from_s(0.0)));
  CHECK(std::isnan(ld::vonmises_kappa_from_s(-1.0)));
  CHECK(std::isnan(ld::vonmises_kappa_from_s_dk<1>(0.0)));
  CHECK(std::isnan(ld::vonmises_kappa_from_s_dk<6>(-1.0)));
  CHECK(std::isnan(lm::bessel_i_ratio_inv_c(-0.5)));
  CHECK(std::isnan(lm::bessel_i_ratio_inv_c(1.5)));
  CHECK(lm::bessel_i_ratio_inv_c(1.0) == 0.0);
  CHECK(std::isinf(lm::bessel_i_ratio_inv_c(0.0)));
  double out[7];
  ld::vonmises_kappa_from_s_upto<6>(-1.0, out);
  CHECK(std::isnan(out[0]));
  CHECK(std::isnan(out[6]));
}
