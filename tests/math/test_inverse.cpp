// Laplace's own inverse incomplete gamma and beta functions
// (include/laplace/math/incomplete_inverse.hpp) against the mpmath references
// of tools/reference/incomplete.py and inverse.py, evaluated in a Kokkos
// kernel on the default execution space.
//
// The error is measured as in benchmarks/math_inverse.cpp: ulpC = |x - x*| /
// max(ulp(x*), dx), dx = ulp(input) / |d input / dx| the change of the exact
// inverse x* caused by one ulp of the input (p, q, log p or log q). It bounds
// the forward error where the problem is well conditioned (dx below an ulp of
// x*) and the backward error where it is not (x moves 1/a ulps per ulp of p
// for a small shape a). The bounds are the errors measured on 2026-11-09 on
// the gcc/OpenMP and CUDA builds with a margin of about half again; they are
// set by the errors of the forward functions own::gamma_inc and own::beta_inc
// (docs/numerics.md), and should come down when those improve.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/incomplete_inverse.hpp>

#include <cmath>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace own = laplace::math::own;

enum class Kind { gp, gq, gpl, gql, bp, bq, bpl, bql };

// laplace::math:: wrappers for x; own:: for 1 - x of the beta (complement)
struct Inverse {
  Kind k;
  bool complement = false;
  KOKKOS_INLINE_FUNCTION double operator()(double a0, double a1, double a2) const {
    switch (k) {
      case Kind::gp: return laplace::math::gamma_p_inv(a0, a1);
      case Kind::gq: return laplace::math::gamma_q_inv(a0, a1);
      case Kind::gpl: return laplace::math::gamma_p_inv_logp(a0, a1);
      case Kind::gql: return laplace::math::gamma_q_inv_logq(a0, a1);
      default: break;
    }
    if (!complement) {
      switch (k) {
        case Kind::bp: return laplace::math::ibeta_inv(a0, a1, a2);
        case Kind::bq: return laplace::math::ibetac_inv(a0, a1, a2);
        case Kind::bpl: return laplace::math::ibeta_inv_logp(a0, a1, a2);
        default: return laplace::math::ibetac_inv_logq(a0, a1, a2);
      }
    }
    switch (k) {
      case Kind::bp: return own::ibeta_inv(a0, a1, a2).y;
      case Kind::bq: return own::ibetac_inv(a0, a1, a2).y;
      case Kind::bpl: return own::ibeta_inv_logp(a0, a1, a2).y;
      default: return own::ibetac_inv_logq(a0, a1, a2).y;
    }
  }
};

bool is_gamma(Kind k) { return k == Kind::gp || k == Kind::gq || k == Kind::gpl || k == Kind::gql; }

// |d input / dx| at the pair (x, y = 1 - x), in double: a yardstick
double sensitivity(Kind k, const bench::Args& arg, double x, double y) {
  if (is_gamma(k)) {
    const double a = arg[0];
    const own::detail_inverse::gamma_shape sh(a);
    double l = sh.log_xdens(x) - std::log(x);
    if (k == Kind::gpl) l -= own::gamma_inc(a, x).log_p;
    if (k == Kind::gql) l -= own::gamma_inc(a, x).log_q;
    return std::exp(l);
  }
  const double a = arg[0], b = arg[1];
  const own::detail_inverse::beta_shape sh(a, b);
  double l = sh.log_d(x, y) - std::log(x) - std::log(y);
  const auto r = x <= 0.5 ? own::beta_inc(a, b, x) : own::beta_inc(b, a, y);
  if (k == Kind::bpl) l -= x <= 0.5 ? r.log_i : r.log_ic;
  if (k == Kind::bql) l -= x <= 0.5 ? r.log_ic : r.log_i;
  return std::exp(l);
}

void check(const char* name, Kind k, double max_c, bool complement = false, double max_c_complement = 0) {
  for (int pass = 0; pass < (complement ? 2 : 1); ++pass) {
    // pass 1: 1 - x against <name>_c.csv, the rows with x* > 1/2
    const bool comp = pass == 1;
    const auto r = bench::read_ref(refdir + "/" + name + (comp ? "_c.csv" : ".csv"));
    const auto y = bench::eval_kernel(r.x, Inverse{k, comp});
    const int in = r.nargs - 1;
    std::size_t bad = 0;
    double worst = 0;
    bench::Args warg{0, 0, 0};
    for (std::size_t i = 0; i < y.size(); ++i) {
      const double e = bench::error_in(y[i], r.hi[i], r.lo[i], r.hi[i]);
      if (std::isinf(e)) {
        ++bad;
        continue;
      }
      const double px = comp ? 1.0 - r.hi[i] : r.hi[i], py = comp ? r.hi[i] : 1.0 - r.hi[i];
      const double dx = bench::ulp_of(r.x[i][in]) / sensitivity(k, r.x[i], px, py);
      const double c =
          std::fabs((y[i] - r.hi[i]) - r.lo[i]) / std::fmax(bench::ulp_of(r.hi[i]), std::isfinite(dx) ? dx : 0.0);
      if (c > worst) {
        worst = c;
        warg = r.x[i];
      }
    }
    const std::string label = std::string(name) + (comp ? " (1 - x)" : "");
    INFO(label, ": max ulpC ", worst, " at (", warg[0], ", ", warg[1], ", ", warg[2], ")");
    CHECK(bad == 0);
    CHECK(worst <= (comp ? max_c_complement : max_c));
  }
}

}  // namespace

// Measured maxima of ulpC (gcc/OpenMP, CUDA): gamma_p_inv 52.2, 52.2;
// gamma_p_inv_tail 657, 657 (P solved to about |log p| ulps: own::gamma_inc
// forms P as exp(log P)); gamma_q_inv 25, 18.1; gamma_q_inv_tail 10.1, 15.2;
// gamma_p_inv_logp 39, 39; gamma_q_inv_logq 11.1, 11.1.
TEST_CASE("inverse incomplete gamma against mpmath") {
  check("gamma_p_inv", Kind::gp, 80.0);
  check("gamma_p_inv_tail", Kind::gp, 1000.0);
  check("gamma_q_inv", Kind::gq, 40.0);
  check("gamma_q_inv_tail", Kind::gq, 23.0);
  check("gamma_p_inv_logp", Kind::gpl, 60.0);
  check("gamma_q_inv_logq", Kind::gql, 17.0);
}

// Measured maxima of ulpC for x and for 1 - x (gcc/OpenMP; CUDA): ibeta_inv
// 483, 2160; 381, 2880. ibeta_inv_tail 827, 423; 827, 513. ibetac_inv 3250,
// 150; 4620, 87. ibetac_inv_tail 388, 918; 1360, 918. ibeta_inv_logp 1310,
// 646; 1220, 667. ibetac_inv_logq 1080, 1620; 1180, 1620. The largest come
// from own::beta_inc where one shape is small and the other large.
TEST_CASE("inverse incomplete beta against mpmath") {
  check("ibeta_inv", Kind::bp, 750.0, true, 4300.0);
  check("ibeta_inv_tail", Kind::bp, 1250.0, true, 800.0);
  check("ibetac_inv", Kind::bq, 7000.0, true, 230.0);
  check("ibetac_inv_tail", Kind::bq, 2050.0, true, 1400.0);
  check("ibeta_inv_logp", Kind::bpl, 2000.0, true, 1000.0);
  check("ibetac_inv_logq", Kind::bql, 1800.0, true, 2450.0);
}

namespace {
struct Special {
  KOKKOS_INLINE_FUNCTION double operator()(double i) const {
    namespace m = laplace::math;
    const double inf = m::detail::pos_inf();
    switch (static_cast<int>(i)) {
      case 0: return m::gamma_p_inv(2.5, 0.0) == 0.0;
      case 1: return m::gamma_p_inv(2.5, 1.0) == inf;
      case 2: return m::gamma_q_inv(2.5, 0.0) == inf;
      case 3: return m::gamma_q_inv(2.5, 1.0) == 0.0;
      case 4: return m::gamma_p_inv_logp(2.5, 0.0) == inf;
      case 5: return m::gamma_p_inv_logp(2.5, -inf) == 0.0;
      case 6: return m::gamma_q_inv_logq(2.5, 0.0) == 0.0;
      case 7: return m::gamma_q_inv_logq(2.5, -inf) == inf;
      case 8: return m::gamma_p_inv(-1.0, 0.5) != m::gamma_p_inv(-1.0, 0.5);  // NaN
      case 9: return m::gamma_p_inv(2.5, 1.5) != m::gamma_p_inv(2.5, 1.5);
      case 10: return m::gamma_p_inv_logp(2.5, 0.1) != m::gamma_p_inv_logp(2.5, 0.1);
      case 11: return m::ibeta_inv(2.0, 3.0, 0.0) == 0.0 && m::ibeta_inv(2.0, 3.0, 1.0) == 1.0;
      case 12: return m::ibetac_inv(2.0, 3.0, 0.0) == 1.0 && m::ibetac_inv(2.0, 3.0, 1.0) == 0.0;
      case 13: return m::ibeta_inv_logp(2.0, 3.0, 0.0) == 1.0 && m::ibeta_inv_logp(2.0, 3.0, -inf) == 0.0;
      case 14: return m::ibeta_inv(2.0, -3.0, 0.5) != m::ibeta_inv(2.0, -3.0, 0.5);
      // roots below 2^-1075 round to 0: P(1/2, x) = 2^-1074 at x ~ 1e-647
      case 15: return m::gamma_p_inv(0.5, 4.9406564584124654e-324) == 0.0;
      // a = 1: P(1, x) = 1 - e^-x, x = p for the smallest subnormal p
      case 16: return m::gamma_p_inv(1.0, 4.9406564584124654e-324) == 4.9406564584124654e-324;
      // the uniform: I_x(1, 1) = x, to 2 ulps
      case 17:
        return Kokkos::fabs(m::ibeta_inv(1.0, 1.0, 0.3) - 0.3) <= 1.2e-16 &&
               Kokkos::fabs(m::ibetac_inv(1.0, 1.0, 0.25) - 0.75) <= 2.3e-16;
      // the complement near 1: 1 - I_x(1, 1) = 1 - x = 1e-20 gives x = 1 and
      // 1 - x = 1e-20 to a relative 1e-15 (the forward function's error)
      case 18: {
        const auto r = laplace::math::own::ibetac_inv(1.0, 1.0, 1e-20);
        return r.x == 1.0 && Kokkos::fabs(r.y - 1e-20) <= 1e-35;
      }
      default: return 0.0;
    }
  }
};
}  // namespace

TEST_CASE("inverse incomplete gamma and beta: special values") {
  std::vector<bench::Args> x;
  for (int i = 0; i <= 18; ++i) x.push_back({double(i), 0, 0});
  const auto y = bench::eval_kernel(x, Special{});
  for (int i = 0; i <= 18; ++i) {
    INFO("special case ", i);
    CHECK(y[i] == 1.0);
  }
}
