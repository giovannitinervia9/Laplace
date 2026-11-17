// box_cox: the Box-Cox bijector, g(v) = (v^lambda - 1)/lambda (log v at lambda = 0), against its mpmath
// references (the shared check of param_common.hpp), plus exact special values, the bounds of the domains,
// the image of lambda = 0 and the limits, invalid parameters, inversion, the distances, the derivatives in
// lambda against finite differences, continuity through lambda = 0, and the limits at the ends of both
// domains (in a kernel on the default space).
// Written 2026-11-12: the bounds below are the measured maxima (ulpC, see the
// docs/numerics.md) times about 1.5.
#include "param_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

namespace lb = laplace::bijector;
using map_t = lb::scalar_bijector_map;

const double inf_ = std::numeric_limits<double>::infinity();
const double nan_ = std::numeric_limits<double>::quiet_NaN();

lb::scalar_bijector box(double lambda) { return lb::scalar_bijector::box_cox(lambda); }

// |a - b| relative to |b| (absolute when b is zero)
double rel(double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-300); }

const double lambdas[] = {-1.5, -0.3, -1e-3, -std::ldexp(1.0, -40), 0.0, std::ldexp(1.0, -40), 1e-3, 0.5, 1.0, 1.7, 3.0};
const double vs[] = {1e-18, 1e-3, 0.02, 0.5, 0.9999, 1.0, 1.0001, 2.0, 7.5, 30.0, 1e6, 1e18};

// the forward value at lambda through a fresh bijector, and the inverse likewise
double g(double lambda, double v) { return box(lambda).forward(v); }
double h(double lambda, double u) { return box(lambda).inverse(u); }

// |a - b| <= rel max(1, |a|, |b|)
bool close_rel(double a, double b, double rel = 1e-14) {
  return std::fabs(a - b) <= rel * std::max({1.0, std::fabs(a), std::fabs(b)});
}

// Exact where either side is not finite, and to `rel` where both are: |inf - inf| is NaN and no
// comparison with a NaN holds, so a tolerance alone would reject two equal infinities. The same
// rule the reference script applies in Python when one side is infinite.
bool same_or_close(double a, double b, double rel) {
  if (!std::isfinite(a) || !std::isfinite(b)) return a == b;
  return close_rel(a, b, rel);
}

double factorial_of(int k) {
  double f = 1.0;
  for (int i = 2; i <= k; ++i) f *= i;
  return f;
}

// the pure derivative of order K, through the member of that order, so that the n1 = 0 column of
// the partials can be compared with the function the engine calls for a pure derivative
double box_cox_pure(const lb::scalar_bijector& b, bool forward, bool log_jac, int K, double x) {
  if (forward && !log_jac)
    switch (K) {
      case 1: return b.forward_d1(x);
      case 2: return b.forward_d2(x);
      case 3: return b.forward_d3(x);
      case 4: return b.forward_d4(x);
      case 5: return b.forward_d5(x);
      default: return b.forward_d6(x);
    }
  if (forward)
    switch (K) {
      case 1: return b.forward_log_jacobian_d1(x);
      case 2: return b.forward_log_jacobian_d2(x);
      case 3: return b.forward_log_jacobian_d3(x);
      case 4: return b.forward_log_jacobian_d4(x);
      case 5: return b.forward_log_jacobian_d5(x);
      default: return b.forward_log_jacobian_d6(x);
    }
  if (!log_jac)
    switch (K) {
      case 1: return b.inverse_d1(x);
      case 2: return b.inverse_d2(x);
      case 3: return b.inverse_d3(x);
      case 4: return b.inverse_d4(x);
      case 5: return b.inverse_d5(x);
      default: return b.inverse_d6(x);
    }
  switch (K) {
    case 1: return b.inverse_log_jacobian_d1(x);
    case 2: return b.inverse_log_jacobian_d2(x);
    case 3: return b.inverse_log_jacobian_d3(x);
    case 4: return b.inverse_log_jacobian_d4(x);
    case 5: return b.inverse_log_jacobian_d5(x);
    default: return b.inverse_log_jacobian_d6(x);
  }
}

}  // namespace

// ---------------------------------------------------------------- the ends of both domains
//
// The quantities at the two ends of the domain of g, which is (0, inf), and at the two ends of
// the domain of h, which is the image of g, evaluated in one kernel on the default execution
// space. For each lambda the four places are: h at the lower end of its domain, h at its upper
// end, g at v = 0 and g at v = +inf; within a place the fourteen quantities are the map of order
// 0 to 6 and then its log-Jacobian of order 0 to 6. The three slots hold the value at the end
// itself and the values at two arguments that approach it, so that the limit can be checked
// against the function's own approach to it. Every end of box_cox has such an approach, the
// finite ends included: there the two arguments are the points where w = 1 + lambda u is 1e-30
// and 1e-60, which lie inside the domain on whichever side of the end it is.
//
// The kernel is a named function with external linkage: an extended device lambda inside a
// doctest TEST_CASE collides with the one of another translation unit (see CLAUDE.md).

constexpr int bc_ends_places = 4;
constexpr int bc_ends_slots = 3;
constexpr int bc_ends_quantities = 14;
// LayoutRight is named and not left to the default: the kernel takes a pointer into the View
// and walks the last index by hand, which is contiguous only in that layout. The default is
// the execution space's own, LayoutRight on the host and LayoutLeft on CUDA, where the FIRST
// index is contiguous: the writes would scatter across lambda and every entry never written
// would read back as the zero Kokkos initialized it with. Measured on CUDA before the layout
// was named: 120 failed assertions in this case, every one of them an infinity read as 0,
// with gcc green throughout. The house pattern is in param_common.hpp.
using bc_ends_view =
    Kokkos::View<double* [bc_ends_places][bc_ends_slots][bc_ends_quantities], Kokkos::LayoutRight>;

namespace {

// o[0..6]: the map of order 0 to 6 at x; o[7..13]: its log-Jacobian of order 0 to 6
KOKKOS_INLINE_FUNCTION void bc_fill_quantities(const lb::scalar_bijector& b, bool forward, double x, double* o) {
  if (forward) {
    o[0] = b.forward(x), o[1] = b.forward_d1(x), o[2] = b.forward_d2(x), o[3] = b.forward_d3(x);
    o[4] = b.forward_d4(x), o[5] = b.forward_d5(x), o[6] = b.forward_d6(x);
    o[7] = b.forward_log_jacobian(x), o[8] = b.forward_log_jacobian_d1(x), o[9] = b.forward_log_jacobian_d2(x);
    o[10] = b.forward_log_jacobian_d3(x), o[11] = b.forward_log_jacobian_d4(x), o[12] = b.forward_log_jacobian_d5(x);
    o[13] = b.forward_log_jacobian_d6(x);
    return;
  }
  o[0] = b.inverse(x), o[1] = b.inverse_d1(x), o[2] = b.inverse_d2(x), o[3] = b.inverse_d3(x);
  o[4] = b.inverse_d4(x), o[5] = b.inverse_d5(x), o[6] = b.inverse_d6(x);
  o[7] = b.inverse_log_jacobian(x), o[8] = b.inverse_log_jacobian_d1(x), o[9] = b.inverse_log_jacobian_d2(x);
  o[10] = b.inverse_log_jacobian_d3(x), o[11] = b.inverse_log_jacobian_d4(x), o[12] = b.inverse_log_jacobian_d5(x);
  o[13] = b.inverse_log_jacobian_d6(x);
}

}  // namespace

void box_cox_ends_kernel(Kokkos::View<const double*> lambda, bc_ends_view out) {
  const double pinf = std::numeric_limits<double>::infinity();  // captured by value
  Kokkos::parallel_for(
      "bijector_box_cox_ends", lambda.extent(0), KOKKOS_LAMBDA(int i) {
        const double lam = lambda(i);
        const auto b = lb::scalar_bijector::box_cox(lam);
        const double small[2] = {1e-30, 1e-60}, big[2] = {1e30, 1e60};
        for (int p = 0; p < bc_ends_places; ++p) {
          const bool forward = p >= 2;
          const bool upper = (p % 2) == 1;
          const double end = forward ? (upper ? pinf : 0.0)
                                     : (upper ? b.inverse_domain_upper() : b.inverse_domain_lower());
          bc_fill_quantities(b, forward, end, &out(i, p, 0, 0));
          for (int s = 0; s < 2; ++s) {
            double x;
            if (forward) {
              x = upper ? big[s] : small[s];  // v runs to +inf, or down to 0
            } else if (Kokkos::isinf(end)) {
              x = upper ? big[s] : -big[s];
            } else {
              x = (small[s] - 1.0) / lam;  // w = 1 + lambda u is small[s], inside the domain
            }
            bc_fill_quantities(b, forward, x, &out(i, p, s + 1, 0));
          }
        }
      });
}

TEST_CASE("box_cox: references") {
  // bound[which][order]: the largest ulpC accepted (which: forward, inverse, forward log-Jacobian,
  // inverse log-Jacobian; the distances are checked exactly below and have no reference rows)
  // measured on the gcc build (2026-11-12): forward 1.61, 3.05, 4.33, 3.99, 4.93, 6.27, 7.97;
  // forward log-Jacobian 1.28, 0.77, 0.67, 0.46, 0.77, 0.64, 0.48. Inverse and inverse
  // log-Jacobian after the refinements of 2026-11-12 (shared log1prel pass, the log-Jacobian's
  // partials from scalar_yeo_johnson.hpp), the larger of gcc and CUDA: inverse 1.12, 4.31, 5.44,
  // 8.76, 13.9, 23, 46.3; inverse log-Jacobian 1.18, 5.22, 2.32, 2.08, 2.76, 3.77, 6.61
  static const double bound[6][7] = {
      {2.5, 5, 6.5, 6, 7.5, 9.5, 12},
      {2, 6.5, 8.5, 13.5, 21, 35, 70},
      {2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5},
      {2, 8, 3.5, 3.5, 4.5, 6, 10},
      {-1, -1, -1, -1, -1, -1, -1},
      {-1, -1, -1, -1, -1, -1, -1}};
  bijector_param_test::check_map("box_cox", map_t::box_cox, bound);
}

TEST_CASE("box_cox: special values") {
  SUBCASE("g(1) = 0 and h(0) = 1 for every lambda") {
    for (double lambda : lambdas) {
      CHECK(g(lambda, 1.0) == 0.0);
      CHECK(h(lambda, 0.0) == 1.0);
    }
  }
  SUBCASE("lambda = 1 is v - 1, the inverse is 1 + u") {
    for (double v : vs)
      if (v < 1e6) CHECK(rel(g(1.0, v), v - 1.0) < 1e-14);
    for (double u : {-0.5, -0.99, 0.3, 3.0, 100.0}) CHECK(rel(h(1.0, u), 1.0 + u) < 1e-14);
  }
  SUBCASE("lambda = 2 is (v^2 - 1)/2") {
    for (double v : {0.5, 3.0, 1000.0}) CHECK(rel(g(2.0, v), 0.5 * (v * v - 1.0)) < 1e-14);
  }
  SUBCASE("lambda = 0 is log v and exp u, exactly as the functions of the reference") {
    for (double v : vs) CHECK(rel(g(0.0, v), std::log(v)) < 1e-15);
    for (double u : {-3.0, 0.5, 20.0, -700.0}) CHECK(rel(h(0.0, u), std::exp(u)) < 1e-15);
  }
  SUBCASE("the first derivative is v^(lambda - 1), and the second is (lambda - 1) v^(lambda - 2)") {
    for (double lambda : {-1.5, 0.5, 1.7}) {
      for (double v : {0.2, 1.0, 5.0}) {
        CHECK(rel(box(lambda).forward_d1(v), std::pow(v, lambda - 1.0)) < 1e-14);
        CHECK(rel(box(lambda).forward_d2(v), (lambda - 1.0) * std::pow(v, lambda - 2.0)) < 1e-14);
      }
    }
  }
  SUBCASE("the mixed derivative d/dv d/dlambda g is v^(lambda - 1) log v; at lambda = 1 it is log v") {
    for (double v : {0.3, 2.0, 40.0}) {
      const auto p = box(1.0).forward_partials_d2(v);
      CHECK(rel(p.c[1], std::log(v)) < 1e-14);
      CHECK(rel(box(0.5).forward_partials_d2(v).c[1], std::pow(v, -0.5) * std::log(v)) < 1e-14);
    }
  }
  SUBCASE("the value on the bounds of the domain of g") {
    CHECK(box(0.5).forward(0.0) == -2.0);           // -1/lambda for lambda > 0
    CHECK(box(-0.5).forward(0.0) == -inf_);         // -inf for lambda < 0
    CHECK(box(0.0).forward(0.0) == -inf_);          // log 0
    CHECK(box(0.5).forward(inf_) == inf_);
    CHECK(box(-0.5).forward(inf_) == 2.0);          // -1/lambda for lambda < 0
    CHECK(box(0.0).forward(inf_) == inf_);
  }
  SUBCASE("the value on the bounds of the image of g (the domain of h)") {
    CHECK(box(0.5).inverse(-2.0) == 0.0);           // h(-1/lambda) = 0 for lambda > 0
    CHECK(box(-0.5).inverse(2.0) == inf_);          // h(-1/lambda) = inf for lambda < 0
  }
  SUBCASE("the domains and images") {
    const auto b = box(0.5);
    CHECK(b.forward_domain_lower() == 0.0);
    CHECK(b.forward_domain_upper() == inf_);
    CHECK(b.inverse_domain_lower() == -2.0);
    CHECK(b.inverse_domain_upper() == inf_);
    const auto c = box(-0.5);
    CHECK(c.inverse_domain_lower() == -inf_);
    CHECK(c.inverse_domain_upper() == 2.0);
    CHECK(box(0.0).inverse_domain_lower() == -inf_);
    CHECK(box(0.0).inverse_domain_upper() == inf_);
  }
  SUBCASE("points outside the domains and NaN parameters give NaN") {
    CHECK(std::isnan(box(0.5).forward(-1.0)));
    CHECK(std::isnan(box(0.5).inverse(-3.0)));     // below -1/lambda
    CHECK(std::isnan(box(-0.5).inverse(3.0)));     // above -1/lambda
    CHECK(std::isnan(box(0.5).forward(nan_)));
    const auto bad = box(nan_);
    CHECK(std::isnan(bad.forward(2.0)));
    CHECK(std::isnan(bad.inverse(1.0)));
    CHECK(std::isnan(bad.forward_d1(2.0)));
    CHECK(std::isnan(bad.inverse_d3(1.0)));
    CHECK(std::isnan(bad.forward_log_jacobian(2.0)));
    CHECK(std::isnan(bad.inverse_log_jacobian_d2(1.0)));
    CHECK(std::isnan(bad.forward_partials_d2(2.0).c[0]));
    CHECK(std::isnan(bad.inverse_partials_d2(1.0).c[1]));
  }
  SUBCASE("log|g'| = (lambda - 1) log v and log|h'| = (1 - lambda) log(1 + lambda u)/lambda") {
    for (double lambda : {-1.5, 0.5, 1.7}) {
      for (double v : {0.2, 3.0})
        CHECK(rel(box(lambda).forward_log_jacobian(v), (lambda - 1.0) * std::log(v)) < 1e-14);
      for (double u : {-0.2, 0.7})
        if (1.0 + lambda * u > 0.0)
          CHECK(rel(box(lambda).inverse_log_jacobian(u), (1.0 - lambda) * std::log1p(lambda * u) / lambda) < 1e-13);
    }
    CHECK(rel(box(0.0).inverse_log_jacobian(0.7), 0.7) < 1e-15);  // log h' = u at lambda = 0
  }
  SUBCASE("the distances: the range of h is (0, inf)") {
    for (double lambda : {-1.5, 0.0, 0.5, 3.0})
      for (double u : {-0.2, 0.7, 4.0}) {
        if (lambda < 0.0 && u >= -1.0 / lambda) continue;  // outside the image
        if (lambda > 0.0 && u <= -1.0 / lambda) continue;
        const auto b = box(lambda);
        CHECK(b.inverse_distance_upper(u) == inf_);
        CHECK(b.inverse_distance_lower(u) == b.inverse(u));
      }
  }
  SUBCASE("static constructor, direction and kinks") {
    const auto b = lb::scalar_bijector::box_cox(0.5);
    CHECK(b.map == map_t::box_cox);
    CHECK(b.exponent == 0.5);
    CHECK(b.inverted == false);
    CHECK(lb::scalar_bijector::box_cox(1.0).map == map_t::box_cox);  // no normalization to identity
    for (double lambda : lambdas) {
      CHECK(box(lambda).is_decreasing() == false);
      CHECK(box(lambda).forward_kinks().count == 0);
      CHECK(box(lambda).inverse_kinks().count == 0);
    }
  }
  SUBCASE("inversion swaps the two directions exactly") {
    for (double lambda : {-1.5, 0.0, 0.5, 3.0}) {
      const auto b = box(lambda);
      auto bi = b;
      bi.inverted = true;
      for (double v : {0.3, 2.0}) {
        CHECK(bi.inverse(v) == b.forward(v));
        CHECK(bi.inverse_d2(v) == b.forward_d2(v));
        CHECK(bi.inverse_log_jacobian_d1(v) == b.forward_log_jacobian_d1(v));
      }
      for (double u : {0.1, -0.2}) {
        if (1.0 + lambda * u <= 0.0) continue;
        CHECK(bi.forward(u) == b.inverse(u));
        CHECK(bi.forward_log_jacobian(u) == b.inverse_log_jacobian(u));
      }
    }
  }
}

TEST_CASE("box_cox: round trip and continuity through lambda = 0") {
  SUBCASE("h(g(v)) = v and g(h(u)) = u") {
    for (double lambda : {-1.5, -0.3, 0.5, 1.7, 3.0}) {
      for (double v : {0.01, 0.5, 2.0, 30.0}) {
        const auto b = box(lambda);
        // near the bound -1/lambda of the image, h(g(v)) is ill-conditioned: skip the points there
        if (1.0 + lambda * b.forward(v) < 1e-3) continue;
        CHECK(rel(b.inverse(b.forward(v)), v) < 1e-11);
      }
      for (double u : {-0.5, 0.0, 1.5}) {
        const auto b = box(lambda);
        if (1.0 + lambda * u <= 0.0) continue;
        CHECK(rel(b.forward(b.inverse(u)), u) < 1e-11 * std::max(1.0, std::fabs(u)));
      }
    }
  }
  SUBCASE("g(v) = log v + lambda (log v)^2 / 2 + ... through lambda = 0") {
    for (double v : {0.02, 0.5, 7.0, 1e9}) {
      const double L = std::log(v);
      for (double lambda : {std::ldexp(1.0, -20), -std::ldexp(1.0, -20)}) {
        // (g - log v)/lambda = (log v)^2/2 + lambda (log v)^3/6 + ...: the next term bounds the error
        const double first = (g(lambda, v) - L) / lambda;
        const double next = std::fabs(lambda) * std::fabs(L * L * L) / 6.0;
        CHECK(std::fabs(first - 0.5 * L * L) <= 2.0 * next + 1e-12 * (1.0 + L * L));
      }
      // g = L exprel(lambda L) = L (1 + lambda L / 2 + ...): the first term is not zero at 2^-40
      const double lam40 = std::ldexp(1.0, -40);
      CHECK(std::fabs(g(lam40, v) - L - 0.5 * lam40 * L * L) <= 1e-12 * std::fabs(L));
    }
  }
  SUBCASE("h(u) = exp(u) (1 + lambda u / 2 ...) through lambda = 0") {
    for (double u : {-2.0, 0.5, 3.0}) {
      // h = exp(u) (1 - lambda u^2/2 + lambda^2 (u^4/8 + u^3/3) + ...)
      const double lambda = std::ldexp(1.0, -20);
      const double first = (h(lambda, u) - std::exp(u)) / lambda;
      const double next = lambda * (std::pow(u, 4) / 8.0 + std::fabs(u * u * u) / 3.0 + 1.0) * std::exp(u);
      CHECK(std::fabs(first + 0.5 * u * u * std::exp(u)) <= 2.0 * next + 1e-12 * std::exp(u));
    }
  }
}

TEST_CASE("box_cox: derivatives in lambda against central differences") {
  // the step 2^-18: the truncation error of the central difference is about 1e-10 relative, the
  // rounding error about 1e-10 relative
  const double step = std::ldexp(1.0, -18);
  for (double lambda : {-0.6, 0.8, 2.2}) {
    const auto b = box(lambda);
    const auto bp = box(lambda + step), bm = box(lambda - step);
    for (double v : {0.3, 2.5, 40.0}) {
      // d/dlambda g and d/dv d/dlambda g
      const double dg = (bp.forward(v) - bm.forward(v)) / (2 * step);
      CHECK(std::fabs(b.forward_partials_d1(v).c[1] - dg) <= 1e-7 * (std::fabs(dg) + 1.0));
      const double dgv = (bp.forward_d1(v) - bm.forward_d1(v)) / (2 * step);
      CHECK(std::fabs(b.forward_partials_d2(v).c[1] - dgv) <= 1e-7 * (std::fabs(dgv) + 1.0));
    }
    for (double u : {-0.1, 0.4, 2.0}) {
      if (1.0 + (lambda + step) * u <= 0.0 || 1.0 + (lambda - step) * u <= 0.0) continue;
      const double dh = (bp.inverse(u) - bm.inverse(u)) / (2 * step);
      CHECK(std::fabs(b.inverse_partials_d1(u).c[1] - dh) <= 1e-7 * (std::fabs(dh) + 1.0));
      const double dhu = (bp.inverse_d1(u) - bm.inverse_d1(u)) / (2 * step);
      CHECK(std::fabs(b.inverse_partials_d2(u).c[1] - dhu) <= 1e-7 * (std::fabs(dhu) + 1.0));
      const double dlj = (bp.inverse_log_jacobian(u) - bm.inverse_log_jacobian(u)) / (2 * step);
      CHECK(std::fabs(b.inverse_log_jacobian_partials_d1(u).c[1] - dlj) <= 1e-7 * (std::fabs(dlj) + 1.0));
    }
  }
}

// ---------------------------------------------------------------- the partials at the ends
//
// The same four places, with the partials of every order instead of the pure derivatives: for
// K = 1 to 6 the K + 1 entries with n2 = 0, out[n1] = d^(K-n1)_x d^n1_lambda, laid out at
// bc_part_at(K) + n1 for a total of 27 per quantity. Slot 0 is the end itself and slot 1 an
// argument that approaches it, which stops at 1e20 for the inverse: the forms there lose entries
// to NaN at a large finite argument, measured from 1e40 up (docs/numerics.md).

constexpr int bc_part_entries = 27;
constexpr int bc_part_quantities = 4;  // g, log|g'|, h, log|h'|
// LayoutRight for the reason given above the pure-derivative view: this kernel walks the last
// two indices by hand through one pointer.
using bc_part_view = Kokkos::View<double* [bc_ends_places][2][bc_part_quantities][bc_part_entries],
                                  Kokkos::LayoutRight>;

KOKKOS_INLINE_FUNCTION constexpr int bc_part_at(int K) { return (K - 1) * (K + 2) / 2; }

namespace {

// the four partials of order K at x, written into the quantity's block of o. The public interface
// has one member per order, so the order is chosen with if constexpr rather than passed.
template <int K>
KOKKOS_INLINE_FUNCTION void bc_fill_partials(const lb::scalar_bijector& b, bool forward, double x, double* o) {
  const int at = bc_part_at(K);
  if (forward) {
    lb::bijector_partials<K> g, lj;
    if constexpr (K == 1) g = b.forward_partials_d1(x), lj = b.forward_log_jacobian_partials_d1(x);
    if constexpr (K == 2) g = b.forward_partials_d2(x), lj = b.forward_log_jacobian_partials_d2(x);
    if constexpr (K == 3) g = b.forward_partials_d3(x), lj = b.forward_log_jacobian_partials_d3(x);
    if constexpr (K == 4) g = b.forward_partials_d4(x), lj = b.forward_log_jacobian_partials_d4(x);
    if constexpr (K == 5) g = b.forward_partials_d5(x), lj = b.forward_log_jacobian_partials_d5(x);
    if constexpr (K == 6) g = b.forward_partials_d6(x), lj = b.forward_log_jacobian_partials_d6(x);
    for (int n1 = 0; n1 <= K; ++n1) {
      o[0 * bc_part_entries + at + n1] = g.c[n1];
      o[1 * bc_part_entries + at + n1] = lj.c[n1];
    }
    return;
  }
  lb::bijector_partials<K> h, lj;
  if constexpr (K == 1) h = b.inverse_partials_d1(x), lj = b.inverse_log_jacobian_partials_d1(x);
  if constexpr (K == 2) h = b.inverse_partials_d2(x), lj = b.inverse_log_jacobian_partials_d2(x);
  if constexpr (K == 3) h = b.inverse_partials_d3(x), lj = b.inverse_log_jacobian_partials_d3(x);
  if constexpr (K == 4) h = b.inverse_partials_d4(x), lj = b.inverse_log_jacobian_partials_d4(x);
  if constexpr (K == 5) h = b.inverse_partials_d5(x), lj = b.inverse_log_jacobian_partials_d5(x);
  if constexpr (K == 6) h = b.inverse_partials_d6(x), lj = b.inverse_log_jacobian_partials_d6(x);
  for (int n1 = 0; n1 <= K; ++n1) {
    o[2 * bc_part_entries + at + n1] = h.c[n1];
    o[3 * bc_part_entries + at + n1] = lj.c[n1];
  }
}

}  // namespace

void box_cox_end_partials_kernel(Kokkos::View<const double*> lambda, bc_part_view out) {
  const double pinf = std::numeric_limits<double>::infinity();
  Kokkos::parallel_for(
      "bijector_box_cox_end_partials", lambda.extent(0), KOKKOS_LAMBDA(int i) {
        const double lam = lambda(i);
        const auto b = lb::scalar_bijector::box_cox(lam);
        for (int p = 0; p < bc_ends_places; ++p) {
          const bool forward = p >= 2;
          const bool upper = (p % 2) == 1;
          const double end = forward ? (upper ? pinf : 0.0)
                                     : (upper ? b.inverse_domain_upper() : b.inverse_domain_lower());
          // The approach to a limit of the partials is slow: a lambda-derivative brings powers of
          // the logarithm of the argument, so the correction decays as v^lambda (log v)^n1 and
          // 1e-20 still leaves 6e-3 relative at lambda = 1/3 and order 6. The forward arguments are
          // therefore taken at the ends of the range, where the forms are clean (measured to
          // 1e300); the inverse ones stop at 1e20, above which the forms lose entries to NaN.
          double near;
          if (forward) near = upper ? 1e300 : 1e-300;
          else if (Kokkos::isinf(end)) near = upper ? 1e20 : -1e20;
          else near = (1e-20 - 1.0) / lam;  // w = 1 + lambda u is 1e-20, inside the domain
          const double xs[2] = {end, near};
          for (int s = 0; s < 2; ++s) {
            double* o = &out(i, p, s, 0, 0);
            bc_fill_partials<1>(b, forward, xs[s], o);
            bc_fill_partials<2>(b, forward, xs[s], o);
            bc_fill_partials<3>(b, forward, xs[s], o);
            bc_fill_partials<4>(b, forward, xs[s], o);
            bc_fill_partials<5>(b, forward, xs[s], o);
            bc_fill_partials<6>(b, forward, xs[s], o);
          }
        }
      });
}

TEST_CASE("box_cox: the limits at the ends of both domains") {
  // lambda: the regimes (below 0, 0, in (0, 1), 1, above 1) and the knife edges where an
  // exponent of a limit vanishes, lambda = 1/n for the inverse and lambda = n for the forward
  // map, n = 1 to 6.
  const std::vector<double> lam = {-1.5,      -1.0, -0.5, -1.0 / 3.0, -0.2, 0.0,  1.0 / 6.0,
                                   0.2,       0.25, 1.0 / 3.0, 0.5,   2.0 / 3.0,  0.7,
                                   1.0,       1.5,  2.0,  2.5,        3.0,  4.0,  5.0, 6.0};
  const int n = int(lam.size());
  Kokkos::View<double*> lam_d("lambda", n);
  auto lam_h = Kokkos::create_mirror_view(lam_d);
  for (int i = 0; i < n; ++i) lam_h(i) = lam[size_t(i)];
  Kokkos::deep_copy(lam_d, lam_h);
  bc_ends_view out_d("ends", n);
  box_cox_ends_kernel(Kokkos::View<const double*>(lam_d), out_d);
  auto out = Kokkos::create_mirror_view(out_d);
  Kokkos::deep_copy(out, out_d);

  // (1) no NaN anywhere, and the returned limit agrees with the function's own approach to the
  // end: a finite limit is reached, an infinite one grows past 1 in its own sign between the two
  // arguments, a zero one decays below 1e-6.
  //
  // Where a power's exponent is tiny but not zero the limit is right and out of reach: the
  // exponent is then about 1e-16 while the logarithm of the argument reaches 745 at most, so the
  // power stays 1 over the whole range of double and the quantity sits on the plateau of the
  // knife edge although its limit is an infinity or a zero. The two arguments give the same value
  // there, which is how a plateau is told from an approach, and all that can be checked is that
  // the plateau is finite and carries the sign of the limit.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    for (int p = 0; p < bc_ends_places; ++p) {
      for (int q = 0; q < bc_ends_quantities; ++q) {
        const double r = out(i, p, 0, q), v1 = out(i, p, 1, q), v2 = out(i, p, 2, q);
        INFO("lambda = ", l, " place ", p, " quantity ", q, ": limit ", r, ", near ", v1, ", nearer ", v2);
        REQUIRE(!std::isnan(r));
        REQUIRE(!std::isnan(v2));
        if (r != inf_ && r != -inf_ && r != 0.0) {
          // the approach to a finite limit of g is only as good as v^lambda at the nearer
          // argument, which is 1e-10 at lambda = 1/6 and v = 1e-60, the slowest of these lambda
          CHECK(close_rel(v2, r, 1e-8));
        } else if (v2 == r) {
          // the limit is already attained: an exact zero, or an overflow to the infinity itself
        } else if (v2 == v1) {
          CHECK(std::isfinite(v2));
          if (r == inf_) CHECK(v2 >= 0.0);
          if (r == -inf_) CHECK(v2 <= 0.0);
        } else if (r == inf_) {
          CHECK(v2 > v1);
          CHECK(v2 > 1.0);
        } else if (r == -inf_) {
          CHECK(v2 < v1);
          CHECK(v2 < -1.0);
        } else {
          CHECK(std::fabs(v2) < std::fabs(v1));
          CHECK(std::fabs(v2) < 1e-6);
        }
      }
    }
  }

  // (2) the values themselves. The range of h is (0, inf), so h is 0 at the lower end of its
  // domain and +inf at the upper one, the finite end included; g at the ends of (0, inf) reaches
  // the two ends of its image, which is what a caller reads the support of a transformed law from.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    INFO("lambda = ", l);
    CHECK(out(i, 0, 0, 0) == 0.0);
    CHECK(out(i, 1, 0, 0) == inf_);
    CHECK(out(i, 2, 0, 0) == box(l).inverse_domain_lower());
    CHECK(out(i, 3, 0, 0) == box(l).inverse_domain_upper());
  }

  auto at = [&](double l, int p, int q) {
    const auto it = std::find(lam.begin(), lam.end(), l);
    REQUIRE(it != lam.end());
    return out(int(it - lam.begin()), p, 0, q);
  };

  // (3) the knife edges, where the exponent of the power vanishes and the prefactor is the whole
  // limit. For h the exponent is 1/lambda - n, which vanishes at lambda = 1/n, and the prefactor
  // is P_n = prod_(r=1..n-1)(1 - r lambda) = (n-1)!/n^(n-1); the edge is reached at both ends of
  // the domain of h, the finite one (place 0, lambda being positive) and u = +inf (place 1).
  const double edges_h[6] = {1.0, 0.5, (2.0 / 3) * (1.0 / 3), 0.75 * 0.5 * 0.25,
                             0.8 * 0.6 * 0.4 * 0.2, (5.0 / 6) * (4.0 / 6) * (3.0 / 6) * (2.0 / 6) * (1.0 / 6)};
  const double edge_lam[6] = {1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0};
  for (int k = 1; k <= 6; ++k) {
    INFO("the edge lambda = 1/", k, " at order ", k);
    CHECK(close_rel(at(edge_lam[k - 1], 0, k), edges_h[k - 1], 1e-15));
    CHECK(close_rel(at(edge_lam[k - 1], 1, k), edges_h[k - 1], 1e-15));
  }
  // for g the exponent is lambda - m, which vanishes at the integer lambda = m, where the
  // prefactor prod_(k=1..m-1)(lambda - k) is (m-1)!; the edge is reached at v = 0 and at v = +inf
  const double edges_g[6] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0};
  for (int m = 1; m <= 6; ++m) {
    INFO("the edge lambda = ", m, " at order ", m);
    CHECK(at(double(m), 2, m) == edges_g[m - 1]);
    CHECK(at(double(m), 3, m) == edges_g[m - 1]);
  }

  // (4) the signs of the infinities, and the zeros.
  // Below 0 the upper end of the image is finite and h runs to +inf there with every order of it,
  // the exponent 1/lambda - n being negative and P_n a product of factors above 1; at u = -inf
  // the same power dies and h and every order of it are 0.
  for (double l : {-1.5, -1.0, -0.5, -1.0 / 3.0, -0.2}) {
    INFO("lambda = ", l);
    for (int q = 0; q <= 6; ++q) {
      CHECK(at(l, 1, q) == inf_);
      CHECK(at(l, 0, q) == 0.0);
    }
  }
  // At lambda = 0 the domain of h is the whole line and h = e^u: every order runs to +inf above
  // and dies below.
  for (int q = 0; q <= 6; ++q) {
    CHECK(at(0.0, 1, q) == inf_);
    CHECK(at(0.0, 0, q) == 0.0);
  }
  // Above 0 the lower end of the image is finite, w goes to 0 from above and the exponent
  // 1/lambda - n decides the other way round: the order is 0 below the edge and an infinity of
  // the sign of P_n above it, 0 where P_n itself vanishes. At lambda = 2/3, where 1/lambda is 1.5,
  // orders 2 and above are past the edge and P_n alternates from n = 3 on.
  CHECK(at(2.0 / 3.0, 0, 1) == 0.0);    // 1/lambda is 1.5, so order 1 is still below the edge
  CHECK(at(2.0 / 3.0, 0, 2) == inf_);   // P_2 = 1/3
  CHECK(at(2.0 / 3.0, 0, 3) == -inf_);  // P_3 = (1/3)(-1/3)
  CHECK(at(2.0 / 3.0, 0, 4) == inf_);
  CHECK(at(0.5, 0, 1) == 0.0);  // 1/lambda = 2, so order 1 is below the edge
  for (int q = 3; q <= 6; ++q) CHECK(at(0.5, 0, q) == 0.0);  // P_q vanishes: h is a quadratic
  for (int q = 1; q <= 5; ++q) CHECK(at(1.0 / 6.0, 0, q) == 0.0);
  // g at v = 0 alternates below the edge, the prefactor being a product of negative factors;
  // at v = +inf it dies below the edge and runs to +inf above it.
  for (double l : {-1.5, 0.0, 0.5, 2.0 / 3.0}) {
    INFO("lambda = ", l);
    for (int q = 1; q <= 6; ++q) CHECK(at(l, 2, q) == (q % 2 ? inf_ : -inf_));
    for (int q = 1; q <= 6; ++q) CHECK(at(l, 3, q) == 0.0);
  }
  for (int q = 1; q <= 2; ++q) CHECK(at(2.5, 3, q) == inf_);  // lambda above the order
  for (int q = 3; q <= 6; ++q) CHECK(at(2.5, 3, q) == 0.0);

  // (5) the log-Jacobians. log|g'| = (lambda - 1) log v runs to an infinity of the sign of
  // 1 - lambda at v = 0 and of lambda - 1 at v = +inf, and log|h'| = (1 - lambda) S runs to one
  // of the sign of 1 - lambda at the upper end of the image for lambda >= 0 and to one of the
  // sign of lambda - 1 at the lower end; below 0 the ends exchange, S running to +inf at the
  // finite upper end and to -inf at u = -inf. At lambda = 1 both maps are the identity and every
  // order of both log-Jacobians is 0. Every argument derivative of a log-Jacobian dies at an
  // infinite end, except that log|h'| is u at lambda = 0 and its first derivative is 1; at a
  // finite end each runs to an infinity of the sign of its own coefficient.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    INFO("lambda = ", l);
    const double s = 1.0 - l;
    if (l == 1.0) {
      for (int p = 0; p < bc_ends_places; ++p)
        for (int q = 7; q <= 13; ++q) CHECK(out(i, p, 0, q) == 0.0);
      continue;
    }
    CHECK(out(i, 2, 0, 7) == (s > 0 ? inf_ : -inf_));   // log|g'| at v = 0
    CHECK(out(i, 3, 0, 7) == (s > 0 ? -inf_ : inf_));   // at v = +inf
    for (int m = 1; m <= 6; ++m) {
      CHECK(out(i, 3, 0, m + 7) == 0.0);  // every v-derivative dies at v = +inf
      // at v = 0 the m-th derivative is (lambda - 1)(-1)^(m-1)(m-1)! v^(-m)
      const double c = -s * ((m % 2) ? 1.0 : -1.0);
      CHECK(out(i, 2, 0, m + 7) == (c > 0.0 ? inf_ : -inf_));
    }
    // log|h'|: place 1 is u = +inf for lambda >= 0 and the finite end for lambda < 0
    CHECK(out(i, 1, 0, 7) == (l >= 0.0 ? (s > 0 ? inf_ : -inf_) : inf_));
    CHECK(out(i, 0, 0, 7) == (l > 0.0 ? (s > 0 ? -inf_ : inf_) : -inf_));
    if (l >= 0.0) {  // u = +inf: the derivatives die, apart from the slope 1 at lambda = 0
      CHECK(out(i, 1, 0, 8) == (l == 0.0 ? 1.0 : 0.0));
      for (int q = 9; q <= 13; ++q) CHECK(out(i, 1, 0, q) == 0.0);
    }
    if (l <= 0.0) {  // u = -inf: the same
      CHECK(out(i, 0, 0, 8) == (l == 0.0 ? 1.0 : 0.0));
      for (int q = 9; q <= 13; ++q) CHECK(out(i, 0, 0, q) == 0.0);
    }
  }
  // at a finite end every order of log|h'| is infinite, with the sign of
  // (1 - lambda)(-1)^(K-1) lambda^(K-1)
  for (double l : {-1.5, -0.5, 0.25, 0.5, 2.0 / 3.0, 1.5, 2.0, 3.0}) {
    const int p = l < 0.0 ? 1 : 0;
    INFO("lambda = ", l, " at the finite end, place ", p);
    for (int K = 1; K <= 6; ++K) {
      double c = 1.0 - l;
      for (int j = 1; j < K; ++j) c *= -l;
      CHECK(at(l, p, K + 7) == (c > 0.0 ? inf_ : -inf_));
    }
  }
}

TEST_CASE("box_cox: the partials at the ends of both domains") {
  const std::vector<double> lam = {-1.5,      -1.0, -0.5, -1.0 / 3.0, -0.2, 0.0,  1.0 / 6.0,
                                   0.2,       0.25, 1.0 / 3.0, 0.5,   2.0 / 3.0,  0.7,
                                   1.0,       1.5,  2.0,  2.5,        3.0,  4.0,  5.0, 6.0};
  const int n = int(lam.size());
  Kokkos::View<double*> lam_d("lambda", n);
  auto lam_h = Kokkos::create_mirror_view(lam_d);
  for (int i = 0; i < n; ++i) lam_h(i) = lam[size_t(i)];
  Kokkos::deep_copy(lam_d, lam_h);
  bc_part_view out_d("partials", n);
  box_cox_end_partials_kernel(Kokkos::View<const double*>(lam_d), out_d);
  auto out = Kokkos::create_mirror_view(out_d);
  Kokkos::deep_copy(out, out_d);

  auto at = [&](double l, int p, int q, int K, int n1, int slot = 0) {
    const auto it = std::find(lam.begin(), lam.end(), l);
    REQUIRE(it != lam.end());
    return out(int(it - lam.begin()), p, slot, q, bc_part_at(K) + n1);
  };

  // (1) no NaN in any entry, at any end, for any lambda. This is the whole of what the repair
  // claims structurally; every sign and every edge value is checked against SymPy's own limit of
  // the definition by tools/reference/bijector_box_cox_forms.py (check_partials_limits), which the
  // numbers below sample.
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < bc_ends_places; ++p)
      for (int q = 0; q < bc_part_quantities; ++q)
        for (int e = 0; e < bc_part_entries; ++e) {
          INFO("lambda = ", lam[size_t(i)], " place ", p, " quantity ", q, " entry ", e);
          CHECK(!std::isnan(out(i, p, 0, q, e)));
        }

  // (2) the n1 = 0 column is the pure derivative of the same order at the same end, which the two
  // code paths compute separately
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    const auto b = box(l);
    const double ends_f[2] = {0.0, inf_};
    const double ends_i[2] = {b.inverse_domain_lower(), b.inverse_domain_upper()};
    for (int K = 1; K <= 6; ++K) {
      const int e = bc_part_at(K);
      INFO("lambda = ", l, " order ", K);
      // To a few ulp and not bitwise: the two paths form the same prefactor
      // prod_(r=1..K-1)(1 - r lambda) in two different functions, and nvcc contracts 1 - r lambda
      // into an FMA at its own discretion, so the products agree to an ulp on the device while
      // being identical on the host. What the check is for is that the partials' n1 = 0 column is
      // the pure derivative, which an ulp does not weaken; it was `==` and failed on CUDA at the
      // knife edges lambda = 1/5 and 1/6, where the prefactor is 24/625 and 5!/6^5.
      for (int s = 0; s < 2; ++s) {
        const int p = s + 2;  // the forward places
        CHECK(same_or_close(out(i, p, 0, 0, e), box_cox_pure(b, true, false, K, ends_f[s]), 4e-16));
        CHECK(same_or_close(out(i, p, 0, 1, e), box_cox_pure(b, true, true, K, ends_f[s]), 4e-16));
        CHECK(same_or_close(out(i, s, 0, 2, e), box_cox_pure(b, false, false, K, ends_i[s]), 4e-16));
        CHECK(same_or_close(out(i, s, 0, 3, e), box_cox_pure(b, false, true, K, ends_i[s]), 4e-16));
      }
    }
  }

  // (3) the knife edge of h at the finite end, K lambda = 1, where the power is 1 and no logarithm
  // is left: every n1 is finite and the column is P_K(lambda) (-1/lambda^2)^n1, the ratio between
  // consecutive entries being -K^2. The columns are mpmath-free exact rationals.
  const double edge_k1[2] = {1.0, -1.0};
  for (int n1 = 0; n1 <= 1; ++n1) CHECK(close_rel(at(1.0, 0, 2, 1, n1), edge_k1[n1], 1e-14));
  const double edge_k2[3] = {0.5, -2.0, 8.0};
  for (int n1 = 0; n1 <= 2; ++n1) CHECK(close_rel(at(0.5, 0, 2, 2, n1), edge_k2[n1], 1e-14));
  const double edge_k3[4] = {2.0 / 9.0, -2.0, 18.0, -162.0};
  for (int n1 = 0; n1 <= 3; ++n1) CHECK(close_rel(at(1.0 / 3.0, 0, 2, 3, n1), edge_k3[n1], 1e-13));
  const double edge_k5[6] = {24.0 / 625.0, -24.0 / 25.0, 24.0, -600.0, 15000.0, -375000.0};
  for (int n1 = 0; n1 <= 5; ++n1) CHECK(close_rel(at(0.2, 0, 2, 5, n1), edge_k5[n1], 1e-13));
  const double edge_k6[7] = {5.0 / 324.0, -5.0 / 9.0, 20.0, -720.0, 25920.0, -933120.0, 33592320.0};
  for (int n1 = 0; n1 <= 6; ++n1) CHECK(close_rel(at(1.0 / 6.0, 0, 2, 6, n1), edge_k6[n1], 1e-13));

  // (4) the degenerate columns of h at the finite end, lambda = 1/M with M an integer below K:
  // zero at n1 = 0, where h^(K) is identically zero, and infinities of sign (-1)^(n0+M) above it
  CHECK(at(1.0, 0, 2, 2, 0) == 0.0);
  CHECK(at(1.0, 0, 2, 2, 1) == inf_);
  CHECK(at(1.0, 0, 2, 2, 2) == -inf_);
  CHECK(at(1.0, 0, 2, 3, 0) == 0.0);
  CHECK(at(1.0, 0, 2, 3, 1) == -inf_);
  CHECK(at(1.0, 0, 2, 3, 2) == inf_);
  CHECK(at(1.0, 0, 2, 3, 3) == -inf_);
  CHECK(at(0.5, 0, 2, 3, 0) == 0.0);
  CHECK(at(0.5, 0, 2, 3, 1) == inf_);
  CHECK(at(0.5, 0, 2, 3, 2) == -inf_);
  CHECK(at(0.5, 0, 2, 3, 3) == inf_);

  // (4b) the infinities of h's partials at the FINITE end, asserted as exact infinities and not
  // merely as large values. The yeo_johnson session measured the reason this matters: at its own
  // finite end it obtained the infinity by letting w = 1 + lambda u cancel to exactly zero, which
  // the host does (lambda times the rounded -1/lambda rounds back to -1) and CUDA does not, nvcc
  // contracting 1 + lambda u into an FMA that keeps the exact product, so w survived at 4e-17 and
  // the infinity came out as a finite 1e100. box_cox cannot fail that way, its finite end being
  // taken by the guard `u == -1/lambda`, a comparison with the bound rather than a cancellation,
  // after which the limit is a literal infinity; these assertions are what shows it on the device.
  // Below 0 every order and every n1 is an infinity of sign (-1)^n1.
  for (double l : {-1.5, -0.5})
    for (int K = 1; K <= 6; ++K)
      for (int n1 = 0; n1 <= K; ++n1) {
        INFO("lambda = ", l, " order ", K, " n1 ", n1);
        CHECK(at(l, 1, 2, K, n1) == ((n1 % 2) ? -inf_ : inf_));
      }
  // above 0, at lambda = 2/3 where 1/lambda is 1.5, orders 2 and up are past the edge
  CHECK(at(2.0 / 3.0, 0, 2, 2, 0) == inf_);
  CHECK(at(2.0 / 3.0, 0, 2, 2, 1) == -inf_);
  CHECK(at(2.0 / 3.0, 0, 2, 2, 2) == inf_);

  // (4c) the same for the log-Jacobian's partials at the finite end, which is the one path that
  // reaches yeo_johnson_inv_lj_pos_partials, the branch the two maps share. box_cox's guard sits in
  // box_cox_inv_lj_partials, above that call, so the shared function never sees a boundary
  // argument from here; these assertions are what shows the guard firing on the device rather than
  // leaving the protection to be inferred from the other map's behaviour. Below 0 the coefficient
  // (1 - lambda)(-1)^n0 lambda^(n1+1-n0) leaves the sign (-1)^n1 for every order.
  for (double l : {-1.5, -0.5})
    for (int K = 1; K <= 6; ++K)
      for (int n1 = 0; n1 <= K; ++n1) {
        INFO("lambda = ", l, " order ", K, " n1 ", n1);
        CHECK(at(l, 1, 3, K, n1) == ((n1 % 2) ? -inf_ : inf_));
      }

  // (5) log|h'| at lambda = 0, where it is affine in u: the constant (-1)^n1 (n1!)^2 wherever the
  // argument derivatives number one more than those in lambda, the same at both infinite ends
  for (int p = 0; p <= 1; ++p) {
    CHECK(at(0.0, p, 3, 1, 0) == 1.0);    // n0 = 1, n1 = 0
    CHECK(at(0.0, p, 3, 3, 1) == -1.0);   // n0 = 2, n1 = 1
    CHECK(at(0.0, p, 3, 5, 2) == 4.0);    // n0 = 3, n1 = 2
    CHECK(at(0.0, p, 3, 2, 0) == 0.0);    // n0 = 2 > n1 + 1: dies
  }

  // (6) the lambda-derivatives of the end of the image, which is what a caller that models lambda
  // would read a moving support bound from: (-1)^(n1+1) n1!/lambda^(n1+1) where the end is finite
  for (double l : {0.5, 1.0, 2.0, 3.0}) {
    for (int n1 = 1; n1 <= 6; ++n1) {
      const double want = ((n1 % 2) ? 1.0 : -1.0) * factorial_of(n1) / std::pow(l, n1 + 1);
      INFO("lambda = ", l, " n1 = ", n1);
      CHECK(close_rel(at(l, 2, 0, n1, n1), want, 1e-13));  // v = 0, n0 = 0
    }
  }
  for (double l : {-1.5, -0.5}) {
    for (int n1 = 1; n1 <= 6; ++n1) {
      const double want = ((n1 % 2) ? 1.0 : -1.0) * factorial_of(n1) / std::pow(l, n1 + 1);
      INFO("lambda = ", l, " n1 = ", n1);
      CHECK(close_rel(at(l, 3, 0, n1, n1), want, 1e-13));  // v = +inf, n0 = 0
    }
  }

  // (7) log|g'| is linear in lambda: every entry with n1 >= 2 is zero, at both ends
  for (int i = 0; i < n; ++i)
    for (int p = 2; p <= 3; ++p)
      for (int K = 2; K <= 6; ++K)
        for (int n1 = 2; n1 <= K; ++n1) {
          INFO("lambda = ", lam[size_t(i)], " place ", p, " order ", K, " n1 ", n1);
          CHECK(out(i, p, 0, 1, bc_part_at(K) + n1) == 0.0);
        }

  // (8) the approach: where the limit is finite the function reaches it, which is the one check the
  // two large arguments can carry for the partials. An infinite limit is only checked for its sign
  // in (3) to (6) and exhaustively against SymPy in the script.
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < bc_ends_places; ++p)
      for (int q = 0; q < bc_part_quantities; ++q)
        for (int e = 0; e < bc_part_entries; ++e) {
          const double r = out(i, p, 0, q, e), v = out(i, p, 1, q, e);
          if (!std::isfinite(r) || r == 0.0 || std::isnan(v)) continue;
          INFO("lambda = ", lam[size_t(i)], " place ", p, " quantity ", q, " entry ", e, ": limit ", r,
               ", near ", v);
          CHECK(close_rel(v, r, 1e-6));
        }
}

TEST_CASE("box_cox: components in the absent second parameter are zero") {
  const auto p3 = box(0.7).forward_partials_d3(2.0);
  for (int i = 4; i < 10; ++i) CHECK(p3.c[i] == 0.0);
  const auto q4 = box(0.7).inverse_partials_d4(0.3);
  for (int i = 5; i < 15; ++i) CHECK(q4.c[i] == 0.0);
  const auto r5 = box(0.7).inverse_log_jacobian_partials_d5(0.3);
  for (int i = 6; i < 21; ++i) CHECK(r5.c[i] == 0.0);
}
