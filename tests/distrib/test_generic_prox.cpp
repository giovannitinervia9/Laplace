// The generic proximal operator in y of the continuous univariate laws (distrib/generic_prox.hpp,
// distrib/weak_convexity.hpp; note/penalties.md section 5h):
//  - prox_y_by_newton against the mpmath references of tools/reference/generic_prox.py, in a Kokkos
//    kernel, on 20 laws and wrappers (the order of CASES there is the order of law_set below);
//  - the optimality conditions at the computed point: the stationarity residual relative to the
//    size of its terms in the interior, the sign of g at a clamped end;
//  - t = 0 (projection), t rho > 1 and rho = infinity (NaN), NaN in;
//  - fixed_distrib reads its parent's rho and gives the parent's prox bit for bit;
//  - distrib_prox_y prefers the member prox_y where one exists.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/distrib/fixed.hpp>
#include <laplace/distrib/generic_prox.hpp>
#include <laplace/distrib/truncated.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/distrib/beta.hpp>
#include <laplace/distrib/cauchy.hpp>
#include <laplace/distrib/exponential.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gumbel.hpp>
#include <laplace/distrib/invgauss.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/logistic.hpp>
#include <laplace/distrib/lognormal1.hpp>
#include <laplace/distrib/pseudohuber.hpp>
#include <laplace/distrib/skewnormal.hpp>
#include <laplace/distrib/skewt.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/distrib/student_t2.hpp>
#include <laplace/distrib/weibull.hpp>
#include <laplace/distrib/weak_convexity.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace generic_prox_test {

namespace ld = laplace::distrib;

// the laws of tools/reference/generic_prox.py, in its order
struct law_set {
  ld::student_t1_distrib st1{0.5, 2.0, 3.0};
  ld::student_t2_distrib st2{-1.0, 1.5, 5.0};
  ld::cauchy_distrib cau{0.0, 0.7};
  ld::pseudohuber_distrib ph1{0.3, 1.2, 1.0};
  ld::pseudohuber_distrib ph2{0.0, 1.0, 0.01};
  ld::logistic_distrib lgs{1.0, 0.5};
  ld::gumbel_distrib gum{-0.5, 2.0};
  ld::skewnormal_distrib sn1{0.0, 1.0, 0.5};
  ld::skewnormal_distrib sn2{2.0, 0.5, -0.95};
  ld::gaussian1_distrib gau{0.2, 3.0};
  ld::gamma1_distrib ga1{2.0, 1.0 / 3.0};
  ld::gamma1_distrib ga2{1.5, 0.8};
  ld::weibull_distrib wb1{2.0, 2.0};
  ld::weibull_distrib wb2{1.0, 1.3};
  ld::lognormal1_distrib lno{0.2, 0.5};
  ld::invgauss_distrib ivg{1.0, 0.5};
  ld::beta_distrib bet{0.3, 10.0};
  ld::truncated_distrib<ld::student_t1_distrib> tst = ld::truncated(ld::student_t1_distrib{0.0, 1.0, 3.0}, -1.0, 2.0);
  ld::truncated_distrib<ld::gaussian1_distrib> tga = ld::truncated(ld::gaussian1_distrib{0.0, 1.0}, 0.5, 3.0);
  ld::exponential_distrib exd{2.0};
  static constexpr int n_cases = 20;

  // f(law) for the law of case k
  template <class F>
  KOKKOS_INLINE_FUNCTION auto apply(int k, F f) const {
    switch (k) {
      case 0: return f(st1);
      case 1: return f(st2);
      case 2: return f(cau);
      case 3: return f(ph1);
      case 4: return f(ph2);
      case 5: return f(lgs);
      case 6: return f(gum);
      case 7: return f(sn1);
      case 8: return f(sn2);
      case 9: return f(gau);
      case 10: return f(ga1);
      case 11: return f(ga2);
      case 12: return f(wb1);
      case 13: return f(wb2);
      case 14: return f(lno);
      case 15: return f(ivg);
      case 16: return f(bet);
      case 17: return f(tst);
      case 18: return f(tga);
      default: return f(exd);
    }
  }
};

struct prox_fn {
  law_set laws;
  KOKKOS_INLINE_FUNCTION double operator()(double k, double v, double t) const {
    return laws.apply(static_cast<int>(k), [&](const auto& d) { return ld::prox_y_by_newton(d, v, t); });
  }
};

// g(y) = y - v - t l'(y) and the size of its terms, |y| + |v| + t |l'(y)|, and g'(y) = 1 - t l''(y)
struct stationarity {
  double g, size, dg;
};
inline stationarity stationarity_at(const law_set& laws, int k, double y, double v, double t) {
  return laws.apply(k, [&](const auto& d) {
    const double l1 = d.template logpdf_dy_dk<1, 0>(y)[0], l2 = d.template logpdf_dy_dk<2, 0>(y)[0];
    return stationarity{(y - v) - t * l1, std::fabs(y) + std::fabs(v) + t * std::fabs(l1), 1.0 - t * l2};
  });
}
inline double rho_of(const law_set& laws, int k) {
  return laws.apply(k, [](const auto& d) { return ld::distrib_weak_convexity(d); });
}
inline double lower_of(const law_set& laws, int k) {
  return laws.apply(k, [](const auto& d) { return d.support_lower(); });
}
inline double upper_of(const law_set& laws, int k) {
  return laws.apply(k, [](const auto& d) { return d.support_upper(); });
}

// res: [0] t = 0 is the projection on the support, [1] t rho > 1 gives NaN, [2] rho = infinity gives NaN
// (gamma, Weibull and beta with a shape below 1), [3] NaN in gives NaN, [4] negative t gives NaN
void run_special_kernel(double* host) {
  Kokkos::View<double[5]> res("generic_prox_special");
  const law_set laws;
  Kokkos::parallel_for(
      "generic_prox_special", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        double bad[5] = {0, 0, 0, 0, 0};
        const double nan = Kokkos::Experimental::quiet_NaN_v<double>;
        const double vs[5] = {-3.0, -0.2, 0.0, 0.7, 40.0};
        for (int k = 0; k < law_set::n_cases; ++k)
          for (double v : vs)
            laws.apply(k, [&](const auto& d) {
              const double lo = d.support_lower(), hi = d.support_upper();
              if (ld::prox_y_by_newton(d, v, 0.0) != (v < lo ? lo : (v > hi ? hi : v))) bad[0] += 1;
              const double rho = ld::distrib_weak_convexity(d);
              if (rho > 0.0 && !Kokkos::isnan(ld::prox_y_by_newton(d, v, 1.01 / rho))) bad[1] += 1;
              if (!Kokkos::isnan(ld::prox_y_by_newton(d, nan, 1.0))) bad[3] += 1;
              if (!Kokkos::isnan(ld::prox_y_by_newton(d, v, -1.0))) bad[4] += 1;
              return 0;
            });
        const ld::gamma1_distrib g{2.0, 2.0};       // shape 1/2
        const ld::weibull_distrib w{1.0, 0.7};      // shape 0.7
        const ld::beta_distrib b{0.05, 10.0};       // a = 0.5
        for (double v : vs) {
          if (!Kokkos::isnan(ld::prox_y_by_newton(g, v, 1e-6))) bad[2] += 1;
          if (!Kokkos::isnan(ld::prox_y_by_newton(w, v, 1e-6))) bad[2] += 1;
          if (!Kokkos::isnan(ld::prox_y_by_newton(b, v, 1e-6))) bad[2] += 1;
        }
        for (int i = 0; i < 5; ++i) res(i) = bad[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 5; ++i) host[i] = h(i);
}

}  // namespace generic_prox_test

// The error is read in ulps of x* divided by the condition number of the problem,
// kappa = (|x*| + |v| + t |l'(x*)|) / (|x*| g'(x*)), g' = 1 - t l'': rounding the terms of
// g(x) = x - v - t l'(x) at their own size moves x* by kappa ulps. A point at an end of the
// support must be the end exactly.
TEST_CASE("generic prox: against mpmath, in a kernel") {
  using namespace generic_prox_test;
  const auto r = bench::read_ref(std::string(LAPLACE_REFERENCE_DIR) + "/generic_prox.csv");
  REQUIRE(r.x.size() > 2000);
  const auto y = bench::eval_kernel(r.x, prox_fn{});
  const law_set laws;
  double worst_cond = 0.0, worst_plain = 0.0;
  std::size_t at = 0, wrong_ends = 0;
  std::vector<double> worst_case(law_set::n_cases, 0.0);
  for (std::size_t i = 0; i < y.size(); ++i) {
    const int k = static_cast<int>(r.x[i][0]);
    const double v = r.x[i][1], t = r.x[i][2], x = r.hi[i];
    if (x == lower_of(laws, k) || x == upper_of(laws, k) || y[i] == lower_of(laws, k) || y[i] == upper_of(laws, k)) {
      if (x != y[i]) {
        ++wrong_ends;
        if (std::getenv("LAPLACE_PRINT_ERRORS"))
          std::printf("  wrong end: case %d, v = %.17g, t = %.17g: %.17g against %.17g\n", k, v, t, y[i], x);
      }
      continue;
    }
    const stationarity s = stationarity_at(laws, k, x, v, t);
    const double kappa = s.size / (std::fabs(x) * s.dg);
    const double e = std::fabs(bench::error_in(y[i], r.hi[i], r.lo[i], x));
    worst_plain = std::fmax(worst_plain, e);
    const double ek = e / std::fmax(1.0, kappa);
    worst_case[k] = std::fmax(worst_case[k], ek);
    if (ek > worst_cond) {
      worst_cond = ek;
      at = i;
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    std::printf("generic_prox: %zu rows, max %.4g ulp, max %.4g ulp/kappa at (case, v, t) = (%g, %.17g, %.17g), wrong ends %zu\n",
                y.size(), worst_plain, worst_cond, r.x[at][0], r.x[at][1], r.x[at][2], wrong_ends);
    for (int k = 0; k < law_set::n_cases; ++k) std::printf("  case %2d: max %.4g ulp/kappa\n", k, worst_case[k]);
  }
  INFO("max ", worst_cond, " ulp/kappa at (", r.x[at][0], ", ", r.x[at][1], ", ", r.x[at][2], ")");
  CHECK(wrong_ends == 0);
  CHECK(worst_cond <= 8.0);  // measured 5.53 gcc, 3.45 CUDA (plain 19.4 and 18.4 ulp)
}

// Optimality at the computed point, without the reference: in the interior the residual of
// g(x) = x - v - t l'(x) is a few units of eps (|x| + |v| + t |l'(x)| + g'(x) |x|), the rounding of
// its terms plus the change of g over one ulp of x (which dominates next to a singular end: beta
// at x = 1 - 1.4e-9); at a lower end g(end+) >= 0, at an upper end g(end-) <= 0 (the constrained
// minimum of a convex objective).
TEST_CASE("generic prox: optimality conditions at the computed point") {
  using namespace generic_prox_test;
  const auto r = bench::read_ref(std::string(LAPLACE_REFERENCE_DIR) + "/generic_prox.csv");
  const auto y = bench::eval_kernel(r.x, prox_fn{});
  const law_set laws;
  double worst = 0.0;
  std::size_t bad_ends = 0, at = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const int k = static_cast<int>(r.x[i][0]);
    const double v = r.x[i][1], t = r.x[i][2], lo = lower_of(laws, k), hi = upper_of(laws, k);
    if (y[i] == lo || y[i] == hi) {
      const double in = y[i] == lo ? std::nextafter(lo, hi) : std::nextafter(hi, lo);
      const double g = stationarity_at(laws, k, in, v, t).g;
      if (y[i] == lo ? g < 0.0 : g > 0.0) ++bad_ends;
      continue;
    }
    const stationarity s = stationarity_at(laws, k, y[i], v, t);
    const double e = std::fabs(s.g) / ((s.size + s.dg * std::fabs(y[i])) * 0x1p-53);
    if (e > worst) {
      worst = e;
      at = i;
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("generic_prox: max residual %.4g eps (terms + g' |x|) at (case, v, t) = (%g, %.17g, %.17g), x = %.17g\n", worst,
                r.x[at][0], r.x[at][1], r.x[at][2], y[at]);
  CHECK(bad_ends == 0);
  CHECK(worst <= 18.0);  // measured 11.9 gcc, 7.92 CUDA
}

TEST_CASE("generic prox: projection at t = 0, NaN beyond t rho = 1 and for infinite rho, in a kernel") {
  double r[5];
  generic_prox_test::run_special_kernel(r);
  CHECK(r[0] == 0.0);
  CHECK(r[1] == 0.0);
  CHECK(r[2] == 0.0);
  CHECK(r[3] == 0.0);
  CHECK(r[4] == 0.0);
}

TEST_CASE("generic prox: rho readings, wrappers and the member route") {
  namespace ld = laplace::distrib;
  const ld::student_t1_distrib st{0.5, 2.0, 3.0};
  CHECK(ld::distrib_weak_convexity(st) == 4.0 / (8.0 * 3.0 * 4.0));
  CHECK(ld::distrib_weak_convexity(ld::student_t2_distrib{0.0, 1.5, 5.0}) == 6.0 / (8.0 * 3.0 * 2.25));
  CHECK(ld::distrib_weak_convexity(ld::invgauss_distrib{1.0, 0.5}) == 0.125);
  CHECK(ld::distrib_weak_convexity(ld::gamma1_distrib{1.0, 2.0}) == ld::distrib_infinity);
  CHECK(ld::distrib_weak_convexity(ld::gamma1_distrib{1.0, 0.5}) == 0.0);
  // fixed: the parent's rho and the parent's prox, bit for bit
  constexpr auto hold_nu = ld::coord_of<ld::student_t1_distrib>("nu");
  const auto fx = ld::fixed<hold_nu>(st, 3.0);
  static_assert(ld::has_distrib_prox_y<std::remove_const_t<decltype(fx)>>);
  CHECK(ld::distrib_weak_convexity(fx) == ld::distrib_weak_convexity(st));
  for (double v : {-30.0, -1.0, 0.5, 2.0, 1e4})
    for (double t : {1e-3, 0.5, 5.0})
      CHECK(ld::prox_y_by_newton(fx, v, t) == ld::prox_y_by_newton(st, v, t));
  // the member, where it exists, wins over the generic route
  const ld::laplace1_distrib la{0.0, 2.0};
  static_assert(ld::has_prox_y_member<ld::laplace1_distrib> && ld::has_distrib_prox_y<ld::laplace1_distrib>);
  CHECK(ld::distrib_prox_y(la, 3.0, 1.0) == la.prox_y(3.0, 1.0));
  // through univariate_as_multivariate: the member bit for bit where it exists, the generic route otherwise
  {
    const ld::univariate_as_multivariate<ld::laplace1_distrib> ula(la);
    const ld::univariate_as_multivariate<ld::student_t1_distrib> ust(st);
    const double th_la[1] = {2.0}, th_st[2] = {2.0, 3.0}, x0[1] = {0.5}, x1[1] = {0.0};
    for (double v : {-30.0, -1.0, 0.25, 0.5, 2.0, 1e4})
      for (double t : {1e-3, 0.5, 2.0, 5.0}) {
        double out[1], w[1];
        const double vv[1] = {v};
        ula.prox_y(ula.theta_state(th_la, w), vv, x1, t, out, w);
        CHECK(out[0] == la.prox_y(v, t));
        ust.prox_y(ust.theta_state(th_st, w), vv, x0, t, out, w);
        CHECK(out[0] == ld::prox_y_by_newton(st, v, t));
      }
  }
  // laws with a kink in y and no member, and laws without a declared rho, have no generic prox
  static_assert(!ld::has_distrib_prox_y<ld::skewt_distrib>);
  static_assert(ld::has_distrib_prox_y<ld::gaussian1_distrib> && ld::has_distrib_prox_y<ld::student_t1_distrib>);
}
