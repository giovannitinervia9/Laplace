// yeo_johnson (include/laplace/bijector/detail/scalar_yeo_johnson.hpp) against the
// mpmath references (tools/reference/bijector_yeo_johnson.py, through param_common.hpp),
// and exact checks: the special values, the kink at 0, the image for each lambda,
// NaN for an invalid lambda, inversion, the distances, the constructor, the kinks
// and the limits at the ends of both domains (in a kernel on the default space).
// Bounds: 1.5 times the larger of the errors measured on the gcc/OpenMP and CUDA builds on 2026-11-12 (the table
// with LAPLACE_PRINT_ERRORS=1).
#include <doctest/doctest.h>

#include "param_common.hpp"

#include <laplace/bijector/scalar.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

namespace lb = laplace::bijector;
using bmap = lb::scalar_bijector_map;

const double inf = std::numeric_limits<double>::infinity();
const double nan_v = std::numeric_limits<double>::quiet_NaN();

lb::scalar_bijector yj(double lambda) { return lb::scalar_bijector::yeo_johnson(lambda); }

// |a - b| <= rel max(1, |a|, |b|)
bool close(double a, double b, double rel = 1e-14) {
  return std::fabs(a - b) <= rel * std::max({1.0, std::fabs(a), std::fabs(b)});
}

// The same number, or the same to a relative tolerance: exact where either side is not finite,
// a tolerance where both are. Two values the device computed by different routes differ in the
// last bits, nvcc contracting a multiply-add into an FMA at its own discretion, so an exact
// comparison between them is not a platform-independent test. The exact case for an infinity is
// not a nicety: |inf - inf| is NaN and no inequality holds for it, so a tolerance alone would
// reject two equal infinities.
bool same_or_close(double a, double b, double rel = 1e-13) {
  if (!std::isfinite(a) || !std::isfinite(b)) return a == b;
  return close(a, b, rel);
}

}  // namespace

// ---------------------------------------------------------------- the ends of both domains
//
// The quantities at the ends of the forward domain (R) and of the inverse domain (the image
// of g), evaluated in one kernel on the default execution space. For each lambda the four
// places are: h at the lower end of its domain, h at its upper end, g at v = -inf and g at
// v = +inf; within a place the fourteen quantities are the map of order 0 to 6 and then its
// log-Jacobian of order 0 to 6. The three slots hold the value at the end itself and, where
// the end is infinite, the values at two large finite arguments of the same sign, so that
// the limit can be checked against the function's own approach to it.
//
// The kernel is a named function with external linkage: an extended device lambda inside a
// doctest TEST_CASE collides with the one of another translation unit (see CLAUDE.md).

constexpr int yj_ends_places = 4;
constexpr int yj_ends_slots = 3;
constexpr int yj_ends_quantities = 14;
// LayoutRight explicitly: the kernel takes &out(i, p, s, 0) and walks the last index, which is
// contiguous only in that layout, while the default is LayoutLeft on CUDA. Left to the default,
// the device writes land in the wrong slots and the entries never written read back as zero
// (measured on 2026-11-17: every infinity came back 0 on CUDA while gcc was green).
using yj_ends_view = Kokkos::View<double* [yj_ends_places][yj_ends_slots][yj_ends_quantities], Kokkos::LayoutRight>;

// The parameter partials at the same four places. For the total order K = 1 to 6 the entries
// c[index(n1, 0)] of bijector_partials<K>, which are d^(K - n1)_x d^n1_lambda for n1 = 0 to K
// (the map has no second parameter, so n2 is 0 throughout), go at yj_part_offset(K) + n1: 27
// entries for the map, then 27 for its log-Jacobian. The two large arguments are 1e20 and 1e40;
// 1e60 and beyond have a case of their own at the end of this file, the loss there having nothing
// to do with an end of the domain.
constexpr int yj_part_count = 27;
constexpr int yj_parts_quantities = 2 * yj_part_count;
using yj_parts_view =
    Kokkos::View<double* [yj_ends_places][yj_ends_slots][yj_parts_quantities], Kokkos::LayoutRight>;

// KOKKOS_INLINE_FUNCTION as well as constexpr: the kernel calls it, and a host-only constexpr
// called from device code is only an nvcc warning (#20013) while returning nothing usable on
// the GPU, which is the trap CLAUDE.md records for flat_tensors::offset.
KOKKOS_INLINE_FUNCTION constexpr int yj_part_offset(int K) { return (K - 1) * (K + 2) / 2; }

namespace {

// o[0..6]: the map of order 0 to 6 at x; o[7..13]: its log-Jacobian of order 0 to 6
KOKKOS_INLINE_FUNCTION void yj_fill_quantities(const lb::scalar_bijector& b, bool forward, double x, double* o) {
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

// o[yj_part_offset(K) + n1] = d^(K-n1)_x d^n1_lambda of the quantity, for this one K
template <int K>
KOKKOS_INLINE_FUNCTION void yj_one_partials(const lb::scalar_bijector& b, int which, double x, double* o) {
  lb::bijector_partials<K> r;
  if (which == 0) {
    if constexpr (K == 1) r = b.forward_partials_d1(x);
    if constexpr (K == 2) r = b.forward_partials_d2(x);
    if constexpr (K == 3) r = b.forward_partials_d3(x);
    if constexpr (K == 4) r = b.forward_partials_d4(x);
    if constexpr (K == 5) r = b.forward_partials_d5(x);
    if constexpr (K == 6) r = b.forward_partials_d6(x);
  } else if (which == 1) {
    if constexpr (K == 1) r = b.inverse_partials_d1(x);
    if constexpr (K == 2) r = b.inverse_partials_d2(x);
    if constexpr (K == 3) r = b.inverse_partials_d3(x);
    if constexpr (K == 4) r = b.inverse_partials_d4(x);
    if constexpr (K == 5) r = b.inverse_partials_d5(x);
    if constexpr (K == 6) r = b.inverse_partials_d6(x);
  } else if (which == 2) {
    if constexpr (K == 1) r = b.forward_log_jacobian_partials_d1(x);
    if constexpr (K == 2) r = b.forward_log_jacobian_partials_d2(x);
    if constexpr (K == 3) r = b.forward_log_jacobian_partials_d3(x);
    if constexpr (K == 4) r = b.forward_log_jacobian_partials_d4(x);
    if constexpr (K == 5) r = b.forward_log_jacobian_partials_d5(x);
    if constexpr (K == 6) r = b.forward_log_jacobian_partials_d6(x);
  } else {
    if constexpr (K == 1) r = b.inverse_log_jacobian_partials_d1(x);
    if constexpr (K == 2) r = b.inverse_log_jacobian_partials_d2(x);
    if constexpr (K == 3) r = b.inverse_log_jacobian_partials_d3(x);
    if constexpr (K == 4) r = b.inverse_log_jacobian_partials_d4(x);
    if constexpr (K == 5) r = b.inverse_log_jacobian_partials_d5(x);
    if constexpr (K == 6) r = b.inverse_log_jacobian_partials_d6(x);
  }
  for (int n1 = 0; n1 <= K; ++n1) o[yj_part_offset(K) + n1] = r.c[lb::bijector_partials<K>::index(n1, 0)];
}

// the pure derivative of order K of the map (lj false) or of its log-Jacobian, in the forward
// direction or the inverse one: what the entry at n1 = 0 of the partials must equal
double yj_pure_of(const lb::scalar_bijector& b, bool fwd, bool lj, int K, double x) {
  if (!lj)
    return fwd ? (K == 1 ? b.forward_d1(x)
                         : K == 2 ? b.forward_d2(x)
                                  : K == 3 ? b.forward_d3(x)
                                           : K == 4 ? b.forward_d4(x) : K == 5 ? b.forward_d5(x) : b.forward_d6(x))
               : (K == 1 ? b.inverse_d1(x)
                         : K == 2 ? b.inverse_d2(x)
                                  : K == 3 ? b.inverse_d3(x)
                                           : K == 4 ? b.inverse_d4(x) : K == 5 ? b.inverse_d5(x) : b.inverse_d6(x));
  return fwd ? (K == 1 ? b.forward_log_jacobian_d1(x)
                       : K == 2 ? b.forward_log_jacobian_d2(x)
                                : K == 3 ? b.forward_log_jacobian_d3(x)
                                         : K == 4 ? b.forward_log_jacobian_d4(x)
                                                  : K == 5 ? b.forward_log_jacobian_d5(x)
                                                           : b.forward_log_jacobian_d6(x))
             : (K == 1 ? b.inverse_log_jacobian_d1(x)
                       : K == 2 ? b.inverse_log_jacobian_d2(x)
                                : K == 3 ? b.inverse_log_jacobian_d3(x)
                                         : K == 4 ? b.inverse_log_jacobian_d4(x)
                                                  : K == 5 ? b.inverse_log_jacobian_d5(x)
                                                           : b.inverse_log_jacobian_d6(x));
}

KOKKOS_INLINE_FUNCTION void yj_fill_partials(const lb::scalar_bijector& b, int which, double x, double* o) {
  yj_one_partials<1>(b, which, x, o);
  yj_one_partials<2>(b, which, x, o);
  yj_one_partials<3>(b, which, x, o);
  yj_one_partials<4>(b, which, x, o);
  yj_one_partials<5>(b, which, x, o);
  yj_one_partials<6>(b, which, x, o);
}

}  // namespace

void yeo_johnson_ends_kernel(Kokkos::View<const double*> lambda, yj_ends_view out) {
  const double pinf = std::numeric_limits<double>::infinity();  // captured by value
  Kokkos::parallel_for(
      "bijector_yeo_johnson_ends", lambda.extent(0), KOKKOS_LAMBDA(int i) {
        const auto b = lb::scalar_bijector::yeo_johnson(lambda(i));
        const double big[2] = {1e30, 1e60};
        for (int p = 0; p < yj_ends_places; ++p) {
          const bool forward = p >= 2;
          const bool upper = (p % 2) == 1;
          const double end = forward ? (upper ? pinf : -pinf)
                                     : (upper ? b.inverse_domain_upper() : b.inverse_domain_lower());
          yj_fill_quantities(b, forward, end, &out(i, p, 0, 0));
          for (int s = 0; s < 2; ++s) {
            const double x = Kokkos::isinf(end) ? (upper ? big[s] : -big[s]) : end;
            yj_fill_quantities(b, forward, x, &out(i, p, s + 1, 0));
          }
        }
      });
}

void yeo_johnson_ends_partials_kernel(Kokkos::View<const double*> lambda, yj_parts_view out) {
  const double pinf = std::numeric_limits<double>::infinity();  // captured by value
  Kokkos::parallel_for(
      "bijector_yeo_johnson_ends_partials", lambda.extent(0), KOKKOS_LAMBDA(int i) {
        const auto b = lb::scalar_bijector::yeo_johnson(lambda(i));
        const double big[2] = {1e20, 1e40};
        for (int p = 0; p < yj_ends_places; ++p) {
          const bool forward = p >= 2;
          const bool upper = (p % 2) == 1;
          const double end = forward ? (upper ? pinf : -pinf)
                                     : (upper ? b.inverse_domain_upper() : b.inverse_domain_lower());
          for (int s = 0; s < yj_ends_slots; ++s) {
            const double x = s == 0 ? end : (Kokkos::isinf(end) ? (upper ? big[s - 1] : -big[s - 1]) : end);
            yj_fill_partials(b, forward ? 0 : 1, x, &out(i, p, s, 0));
            yj_fill_partials(b, forward ? 2 : 3, x, &out(i, p, s, yj_part_count));
          }
        }
      });
}

TEST_CASE("bijector yeo_johnson: accuracy against mpmath") {
  // 1.5 times the measured maxima of ulpC (LAPLACE_PRINT_ERRORS=1), the larger of gcc/OpenMP and
  // CUDA, after the refinements of 2026-11-12 (the forward log-Jacobian: gcc, earlier that day)
  static const double bound[6][7] = {
      {3, 5, 3, 4, 5, 7, 9},         // forward
      {3, 4, 6, 13, 18, 46, 31},     // inverse
      {2, 2, 2, 4, 4, 4, 5},         // forward log-Jacobian
      {3, 2, 6, 10, 6, 7, 10},       // inverse log-Jacobian
      {-1, -1, -1, -1, -1, -1, -1},  // distances: +inf, no rows
      {-1, -1, -1, -1, -1, -1, -1},
  };
  bijector_param_test::check_map("yeo_johnson", bmap::yeo_johnson, bound);
}

TEST_CASE("bijector yeo_johnson: special values and the kink at 0") {
  for (double lambda : {-0.5, 0.0, 0.4, 1.0, 1.6, 2.0, 2.5}) {
    const auto b = yj(lambda);
    CHECK(b.forward(0.0) == 0.0);
    CHECK(b.forward_d1(0.0) == 1.0);
    CHECK(close(b.forward_d2(0.0), lambda - 1.0));
    // the right-hand third derivative at 0 is (lambda - 1)(lambda - 2)
    CHECK(close(b.forward_d3(0.0), (lambda - 1.0) * (lambda - 2.0)));
    CHECK(b.inverse(0.0) == 0.0);
    CHECK(b.inverse_d1(0.0) == 1.0);
    // the left-hand third derivative (v < 0) is lambda (lambda - 1): they differ unless lambda = 1
    CHECK(close(b.forward_d3(-1e-300), lambda * (lambda - 1.0)));
    CHECK(b.forward_log_jacobian(0.0) == 0.0);
  }
  // lambda = 1 is the identity
  const auto id = yj(1.0);
  for (double v : {-3.0, -0.2, 0.0, 0.7, 40.0}) CHECK(close(id.forward(v), v, 1e-13));
  CHECK(close(id.forward_log_jacobian(0.7), 0.0, 1e-15));
  CHECK(id.forward_log_jacobian_d1(0.7) == 0.0);
  CHECK(id.forward_d2(0.7) == 0.0);
  CHECK(id.forward_d3(-0.7) == 0.0);
  // lambda = 0: g(v) = log(1 + v) on v >= 0, g(v) = v - v^2/2 on v < 0; h(u) = e^u - 1 on u >= 0
  const auto b0 = yj(0.0);
  for (double v : {1e-12, 0.3, 5.0}) CHECK(close(b0.forward(v), std::log1p(v), 1e-14));
  for (double v : {-1e-12, -0.3, -0.9}) CHECK(close(b0.forward(v), v - 0.5 * v * v, 1e-14));
  for (double u : {1e-12, 0.3, 4.0}) CHECK(close(b0.inverse(u), std::expm1(u), 1e-14));
  // lambda = 2: g(v) = -log(1 - v) on v < 0, h(u) = 1 - e^(-u) on u < 0
  const auto b2 = yj(2.0);
  for (double v : {-1e-12, -0.3, -0.9}) CHECK(close(b2.forward(v), -std::log1p(-v), 1e-14));
  for (double u : {-1e-12, -0.3, -3.0}) CHECK(close(b2.inverse(u), -std::expm1(-u), 1e-14));
  // kinks: one at 0, continuous to the second derivative, none at lambda = 1
  for (double lambda : {-0.5, 0.0, 0.4, 1.6, 2.5}) {
    const auto k = yj(lambda).forward_kinks();
    CHECK(k.count == 1);
    CHECK(k.point[0] == 0.0);
    CHECK(k.continuous_order == 2);
    const auto ki = yj(lambda).inverse_kinks();
    CHECK(ki.count == 1);
    CHECK(ki.point[0] == 0.0);
    CHECK(ki.continuous_order == 2);
  }
  CHECK(id.forward_kinks().count == 0);
  CHECK(id.inverse_kinks().count == 0);
}

TEST_CASE("bijector yeo_johnson: image, domain, invalid parameters") {
  // the forward domain is R for every lambda; the image (domain of h) depends on lambda
  for (double lambda : {-2.0, -0.5, 0.0, 1.0, 2.0, 2.5, 4.0}) {
    CHECK(yj(lambda).forward_domain_lower() == -inf);
    CHECK(yj(lambda).forward_domain_upper() == inf);
  }
  CHECK(yj(1.0).inverse_domain_lower() == -inf);
  CHECK(yj(1.0).inverse_domain_upper() == inf);
  CHECK(yj(0.4).inverse_domain_lower() == -inf);
  CHECK(yj(0.4).inverse_domain_upper() == inf);
  // lambda < 0: the positive branch is bounded above by -1/lambda
  CHECK(yj(-0.5).inverse_domain_upper() == 2.0);
  CHECK(yj(-0.5).inverse_domain_lower() == -inf);
  CHECK(std::isnan(yj(-0.5).inverse(2.5)));
  CHECK(std::isfinite(yj(-0.5).inverse(1.99)));
  // lambda > 2: the negative branch is bounded below by 1/(2 - lambda)
  CHECK(yj(2.5).inverse_domain_lower() == -2.0);
  CHECK(yj(2.5).inverse_domain_upper() == inf);
  CHECK(std::isnan(yj(2.5).inverse(-2.5)));
  CHECK(std::isfinite(yj(2.5).inverse(-1.99)));
  // the limits of the image are approached: g(v) -> -1/lambda for lambda < 0, and at large v
  CHECK(std::fabs(yj(-0.5).forward(1e30) - 2.0) < 1e-12);
  CHECK(std::fabs(yj(2.5).forward(-1e30) + 2.0) < 1e-12);
  // invalid lambda: NaN everywhere
  const auto bad = yj(nan_v);
  CHECK(std::isnan(bad.forward(0.3)));
  CHECK(std::isnan(bad.forward_d2(-0.3)));
  CHECK(std::isnan(bad.forward_log_jacobian(0.3)));
  CHECK(std::isnan(bad.inverse(0.3)));
  CHECK(std::isnan(bad.inverse_d3(-0.3)));
  CHECK(std::isnan(bad.inverse_log_jacobian_d1(0.3)));
  CHECK(std::isnan(bad.forward_domain_upper()));
  CHECK(std::isnan(bad.inverse_domain_lower()));
  CHECK(std::isnan(bad.forward_partials_d2(0.3).c[0]));
  CHECK(std::isnan(bad.inverse_partials_d2(0.3).c[1]));
  CHECK(std::isnan(bad.inverse_distance_upper(0.3)));
}

TEST_CASE("bijector yeo_johnson: inversion, distances, direction, constructor") {
  for (double lambda : {-0.5, 0.0, 0.4, 1.6, 2.5}) {
    const auto b = yj(lambda);
    for (double v : {-4.0, -0.3, 0.0, 0.2, 3.0}) {
      const double u = b.forward(v);
      CHECK(close(b.inverse(u), v, 1e-12));
    }
    // an inverted bijector swaps the two directions
    lb::scalar_bijector r = b;
    r.inverted = true;
    CHECK(r.forward(0.7) == b.inverse(0.7));
    CHECK(r.inverse(-0.6) == b.forward(-0.6));
    CHECK(r.forward_d2(-0.6) == b.inverse_d2(-0.6));
    CHECK(r.inverse_log_jacobian(0.7) == b.forward_log_jacobian(0.7));
    // the range of h is R: the distances to its bounds are +inf
    CHECK(b.inverse_distance_upper(0.3) == inf);
    CHECK(b.inverse_distance_lower(0.3) == inf);
    CHECK(!b.is_decreasing());
  }
  CHECK(!yj(-0.5).is_decreasing());
  CHECK(!yj(2.5).is_decreasing());
  // the constructor and the parameter
  const auto c = lb::scalar_bijector::yeo_johnson(0.5);
  CHECK(c.map == bmap::yeo_johnson);
  CHECK(c.exponent == 0.5);
  CHECK(c.parameter_count() == 1);
  CHECK(!c.inverted);
}

TEST_CASE("bijector yeo_johnson: the limits at the ends of both domains") {
  // lambda: the four regimes (below 0, in [0, 2], above 2) and the knife edges where an
  // exponent of the limit vanishes, lambda = 1/n on the positive branch and 2 - 1/n on the
  // negative one, for n = 1 to 6.
  const std::vector<double> lam = {-1.5,        -0.5,        0.0,        1.0 / 6.0,  0.2, 0.25,
                                   1.0 / 3.0,   0.5,         1.0,        1.5,        5.0 / 3.0,
                                   1.75,        1.8,         11.0 / 6.0, 2.0,        2.5, 4.0};
  const int n = int(lam.size());
  Kokkos::View<double*> lam_d("lambda", n);
  auto lam_h = Kokkos::create_mirror_view(lam_d);
  for (int i = 0; i < n; ++i) lam_h(i) = lam[size_t(i)];
  Kokkos::deep_copy(lam_d, lam_h);
  yj_ends_view out_d("ends", n);
  yeo_johnson_ends_kernel(Kokkos::View<const double*>(lam_d), out_d);
  auto out = Kokkos::create_mirror_view(out_d);
  Kokkos::deep_copy(out, out_d);

  // (1) no NaN anywhere, and at an infinite end the returned limit agrees with the function's
  // own approach to it: a finite limit is reached, an infinite one grows past 1 in its own
  // sign between the arguments 1e30 and 1e60, a zero one decays below 1e-6. The two large
  // arguments say nothing at a finite end, where the whole domain lies on one side of the end
  // and the values at the ends themselves are checked in (2) and (4) instead.
  //
  // Where lambda is within an ulp of a knife edge the power's exponent is about 1e-16 while
  // the logarithm of the argument reaches 138 at most, so the power stays 1 over the whole
  // range of double and the quantity sits on the plateau of the edge: the limit is then
  // reached far beyond any representable argument, and all that can be checked is that the
  // plateau is finite and carries the sign of the limit. The two arguments give the same value
  // there, which is how a plateau is told from an approach.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    for (int p = 0; p < yj_ends_places; ++p) {
      // places 2 and 3 are the ends of R; the domain of h is bounded above below lambda = 0
      // and below above lambda = 2
      const bool end_infinite = p >= 2 || (p == 1 ? l >= 0.0 : l <= 2.0);
      for (int q = 0; q < yj_ends_quantities; ++q) {
        const double r = out(i, p, 0, q), v1 = out(i, p, 1, q), v2 = out(i, p, 2, q);
        INFO("lambda = ", l, " place ", p, " quantity ", q, ": limit ", r, ", at 1e30 ", v1, ", at 1e60 ", v2);
        REQUIRE(!std::isnan(r));
        if (!end_infinite) continue;
        if (r != inf && r != -inf && r != 0.0) {
          CHECK(close(v2, r, 1e-12));
        } else if (v2 == r) {
          // the limit is already attained at 1e60: an exact zero, or an overflow to the
          // infinity itself (h is e^u - 1 at lambda = 0 above and 1 - e^(-u) at lambda = 2
          // below, both of which overflow long before 1e30)
        } else if (same_or_close(v2, v1)) {
          CHECK(std::isfinite(v2));
          if (r == inf) CHECK(v2 >= 0.0);
          if (r == -inf) CHECK(v2 <= 0.0);
        } else if (r == inf) {
          CHECK(v2 > v1);
          CHECK(v2 > 1.0);
        } else if (r == -inf) {
          CHECK(v2 < v1);
          CHECK(v2 < -1.0);
        } else {
          CHECK(std::fabs(v2) < std::fabs(v1));
          CHECK(std::fabs(v2) < 1e-6);
        }
      }
    }
  }

  // (2) the values themselves. The range of h is R, so h runs to an infinity of the sign of
  // u at both ends of its domain, the finite one included; g at the ends of R reaches the two
  // ends of its image, which is what a caller reads the support of a transformed law from.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    INFO("lambda = ", l);
    CHECK(out(i, 0, 0, 0) == -inf);
    CHECK(out(i, 1, 0, 0) == inf);
    CHECK(out(i, 2, 0, 0) == yj(l).inverse_domain_lower());
    CHECK(out(i, 3, 0, 0) == yj(l).inverse_domain_upper());
  }

  // (3) the knife edges, where the exponent of the power vanishes and the limit is the
  // prefactor: h^(n) at u = +inf is prod_(k=1..n-1)(1 - k lambda) at lambda = 1/n, and at
  // u = -inf it is prod_(k=1..n-1)(k c - 1) at c = 2 - lambda = 1/n.
  // The edge is taken where the product of lambda (of c) with the order is exactly 1 in double
  // arithmetic: lambda = 1/n is such a double for every n below, while c = 2 - lambda is one
  // only for a dyadic 1/n, so the negative branch has the edge at c = 1, 1/2 and 1/4 and not
  // at 1/3, 1/5 or 1/6, where the limit is then the infinity or the zero of that very lambda.
  auto at = [&](double l, int p, int q) {
    const auto it = std::find(lam.begin(), lam.end(), l);
    return out(int(it - lam.begin()), p, 0, q);
  };
  CHECK(close(at(1.0 / 6.0, 1, 6), (5.0 / 6) * (4.0 / 6) * (3.0 / 6) * (2.0 / 6) * (1.0 / 6)));
  CHECK(close(at(0.2, 1, 5), 0.8 * 0.6 * 0.4 * 0.2));
  CHECK(at(0.25, 1, 4) == 0.75 * 0.5 * 0.25);
  CHECK(close(at(1.0 / 3.0, 1, 3), (2.0 / 3) * (1.0 / 3)));
  CHECK(at(0.5, 1, 2) == 0.5);
  CHECK(at(1.0, 1, 1) == 1.0);  // lambda = 1 is the identity
  CHECK(at(1.0, 0, 1) == 1.0);
  CHECK(at(1.5, 0, 2) == -0.5);                   // c = 1/2
  CHECK(at(1.75, 0, 4) == -0.75 * -0.5 * -0.25);  // c = 1/4
  // the same for the forward map, whose exponent is lambda - n (c - n on the negative
  // branch) and whose prefactor is then (-1)^(n-1) (n - 1)!
  CHECK(at(2.0, 3, 2) == 1.0);
  CHECK(at(4.0, 3, 4) == 6.0);
  CHECK(at(0.0, 2, 2) == -1.0);  // c = 2: -(c - 1) = -1
  CHECK(at(1.0, 2, 1) == 1.0);
  CHECK(at(1.0, 3, 1) == 1.0);

  // (4) the signs of the infinities. At the lower end h^(n) alternates, (-1)^(n-1), while
  // the exponent 1 - n c stays positive; at a finite lower end (lambda > 2, where c < 0) it
  // stays positive for every n, and likewise at a finite upper end (lambda < 0).
  for (double l : {2.5, 4.0}) {
    for (int q = 1; q <= 6; ++q) CHECK(at(l, 0, q) == (q % 2 ? inf : -inf));
    for (int q = 1; q <= 6; ++q) CHECK(at(l, 0, q + 7) == (q % 2 ? -inf : inf));  // log|h'|
  }
  for (double l : {-1.5, -0.5}) {
    for (int q = 0; q <= 6; ++q) CHECK(at(l, 1, q) == inf);
    for (int q = 0; q <= 6; ++q) CHECK(at(l, 1, q + 7) == inf);
  }
  for (int q = 1; q <= 6; ++q) CHECK(at(2.0, 0, q) == (q % 2 ? inf : -inf));  // c = 0: h = 1 - e^(-u)
  CHECK(at(2.0, 0, 8) == -1.0);  // log|h'| = -u there, so its slope is -1 and the rest is 0

  // (5) the log-Jacobians. log|h'| runs to an infinity of the sign of 1 - lambda at u = +inf
  // and of lambda - 1 at u = -inf, and is 0 at lambda = 1. Its u-derivatives die, except at
  // lambda = 0 above and lambda = 2 below, where log|h'| is u and -u.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    INFO("lambda = ", l);
    const double s_up = 1.0 - l, s_lo = l - 1.0;
    CHECK(out(i, 1, 0, 7) == (s_up > 0 ? inf : s_up < 0 ? -inf : 0.0));
    CHECK(out(i, 0, 0, 7) == (s_lo > 0 ? inf : s_lo < 0 ? -inf : 0.0));
    // the u-derivatives die at an infinite end, the slope of log|h'| at the two edges where it
    // is affine apart; at a finite end (lambda < 0 above, lambda > 2 below) they are infinite,
    // and the signs are checked in (4)
    if (l <= 2.0) {
      CHECK(out(i, 0, 0, 8) == (l == 2.0 ? -1.0 : 0.0));
      for (int q = 9; q <= 13; ++q) CHECK(out(i, 0, 0, q) == 0.0);
    }
    if (l >= 0.0) {
      CHECK(out(i, 1, 0, 8) == (l == 0.0 ? 1.0 : 0.0));
      for (int q = 9; q <= 13; ++q) CHECK(out(i, 1, 0, q) == 0.0);
    }
    // log|g'| is (lambda - 1) log(1 + v) above and (1 - lambda) log(1 - v) below, and all its
    // v-derivatives die at both ends
    CHECK(out(i, 3, 0, 7) == (l > 1 ? inf : l < 1 ? -inf : 0.0));
    CHECK(out(i, 2, 0, 7) == (l < 1 ? inf : l > 1 ? -inf : 0.0));
    for (int q = 8; q <= 13; ++q) {
      CHECK(out(i, 2, 0, q) == 0.0);
      CHECK(out(i, 3, 0, q) == 0.0);
    }
  }
}

TEST_CASE("bijector yeo_johnson: the parameter partials at the ends of both domains") {
  // The same lambdas and the same four places as the case above, now for the mixed derivatives
  // in the argument and in lambda. The limits and their derivation are in docs/numerics.md,
  // and every one of them was compared with SymPy's limit of the definition in
  // tools/reference/bijector_yeo_johnson_forms.py (check_partial_limits, check_edge_limits).
  const std::vector<double> lam = {-1.5,      -0.5, 0.0,  1.0 / 6.0, 0.2,        0.25, 1.0 / 3.0, 0.5, 1.0,
                                   1.5,  5.0 / 3.0, 1.75, 1.8,       11.0 / 6.0, 2.0,  2.5,       4.0};
  const int n = int(lam.size());
  Kokkos::View<double*> lam_d("lambda", n);
  auto lam_h = Kokkos::create_mirror_view(lam_d);
  for (int i = 0; i < n; ++i) lam_h(i) = lam[size_t(i)];
  Kokkos::deep_copy(lam_d, lam_h);
  yj_parts_view out_d("ends_partials", n);
  yeo_johnson_ends_partials_kernel(Kokkos::View<const double*>(lam_d), out_d);
  auto out = Kokkos::create_mirror_view(out_d);
  Kokkos::deep_copy(out, out_d);

  // (1) no NaN anywhere; the entry at n1 = 0 is the pure derivative of order K, which the case
  // above pins against the mathematics; and at an infinite end the limit agrees with the
  // function's own approach to it, a plateau or a limit already attained apart (as above). The
  // decay of a vanishing mixed derivative carries powers of the logarithm of the argument, so
  // it is slow: it is checked as a halving between 1e20 and 1e40 and not against a threshold.
  for (int i = 0; i < n; ++i) {
    const double l = lam[size_t(i)];
    const auto b = yj(l);
    for (int p = 0; p < yj_ends_places; ++p) {
      const bool end_infinite = p >= 2 || (p == 1 ? l >= 0.0 : l <= 2.0);
      const double end = p == 2 ? -inf : p == 3 ? inf : (p == 1 ? b.inverse_domain_upper() : b.inverse_domain_lower());
      for (int K = 1; K <= 6; ++K) {
        for (int n1 = 0; n1 <= K; ++n1) {
          for (int lj = 0; lj < 2; ++lj) {
            const int q = yj_part_offset(K) + n1 + lj * yj_part_count;
            const double r = out(i, p, 0, q), v1 = out(i, p, 1, q), v2 = out(i, p, 2, q);
            INFO("lambda = ", l, " place ", p, " K = ", K, " n1 = ", n1);
            INFO("quantity ", (lj ? "log-Jacobian" : "map"), ": limit ", r, ", at 1e20 ", v1, ", at 1e40 ", v2);
            REQUIRE(!std::isnan(r));
            if (n1 == 0) {
              const double pure = yj_pure_of(b, p >= 2, lj == 1, K, end);
              CHECK((r == pure || close(r, pure, 1e-13)));  // r == pure also for two infinities
            }
            if (!end_infinite) continue;
            if (r != inf && r != -inf && r != 0.0) {
              CHECK(close(v2, r, 1e-6));
            } else if (v2 == r) {
              // the limit is already attained at 1e40: an exact zero, or an overflow to it
            } else if (same_or_close(v2, v1)) {
              CHECK(std::isfinite(v2));
              if (r == inf) CHECK(v2 >= 0.0);
              if (r == -inf) CHECK(v2 <= 0.0);
            } else if (r == inf) {
              CHECK(v2 > v1);
              CHECK(v2 > 1.0);
            } else if (r == -inf) {
              CHECK(v2 < v1);
              CHECK(v2 < -1.0);
            } else {
              CHECK(std::fabs(v2) < 0.5 * std::fabs(v1));
            }
          }
        }
      }
    }
  }

  // (2) the written-out values. At an end of R the forward map reaches an end of its image,
  // so its pure lambda-derivatives there are those of -1/lambda below lambda = 0 and of
  // 1/(2 - lambda) above lambda = 2, that is (-1)^(K+1) K!/lambda^(K+1) and K!/(2 - lambda)^(K+1).
  auto at = [&](double l, int p, int K, int n1, int lj) {
    const auto it = std::find(lam.begin(), lam.end(), l);
    return out(int(it - lam.begin()), p, 0, yj_part_offset(K) + n1 + lj * yj_part_count);
  };
  double fact = 1.0;
  for (int K = 1; K <= 6; ++K) {
    fact *= K;
    for (double l : {-1.5, -0.5}) {
      INFO("lambda = ", l, " K = ", K);
      CHECK(close(at(l, 3, K, K, 0), (K % 2 ? 1.0 : -1.0) * fact / std::pow(l, K + 1), 1e-12));
    }
    for (double l : {2.5, 4.0}) {
      INFO("lambda = ", l, " K = ", K);
      CHECK(close(at(l, 2, K, K, 0), fact / std::pow(2.0 - l, K + 1), 1e-12));
    }
  }
  // log|h'| is u at lambda = 0 above and -u at lambda = 2 below up to higher powers of lambda,
  // so where the number of argument derivatives is one more than the number in lambda the mixed
  // derivative is the constant (-1)^n1 (n1!)^2 above, times (-1)^n0 below
  for (int n1 = 0; 2 * n1 + 1 <= 6; ++n1) {
    const int K = 2 * n1 + 1;  // n0 = n1 + 1
    double f = 1.0;
    for (int k = 2; k <= n1; ++k) f *= k;
    INFO("K = ", K, " n1 = ", n1);
    CHECK(close(at(0.0, 1, K, n1, 1), (n1 % 2 ? -1.0 : 1.0) * f * f, 1e-12));
    CHECK(close(at(2.0, 0, K, n1, 1), ((n1 + 1) % 2 ? -1.0 : 1.0) * f * f, 1e-12));
  }
  // At the finite end of the domain of h every mixed derivative is infinite, for the map as
  // well as for its log-Jacobian: h^(n0) = P_n0 w^(1/lambda - n0) with w vanishing and the
  // exponent negative, and each lambda-derivative multiplies by a term dominated by
  // (1/lambda - n0) u/w. The signs are (-1)^n1 and (-1)^(n0+1) for the map, (-1)^n1 and
  // (-1)^n0 for the log-Jacobian, at the upper end (lambda < 0) and the lower one (lambda > 2).
  //
  // These are asserted as exact infinities and not merely as large values on purpose, and the
  // rule has two halves. Both quantities reach that end through a guard; without one they would
  // come from the forms, where w = 1 + lambda u cancels to exactly zero on the host and survives
  // at about 4e-17 under FMA contraction on CUDA, so the infinity would read back as a finite
  // 1e100. A check for "not NaN", for "large" or for "finite or infinite" passes on both
  // platforms whether the guard fires or not, so it cannot tell a working guard from a missing
  // one; only the exact comparison can. The second half is that the exact comparison can do it
  // only ON THE DEVICE: remove the guard and on gcc the forms return the very same infinity,
  // the cancellation being exact there, so these lines stay green on the host either way. A
  // host run says nothing about whether a guard at a finite end is exercised.
  for (double l : {-1.5, -0.5}) {
    for (int K = 1; K <= 6; ++K)
      for (int n1 = 0; n1 <= K; ++n1) {
        INFO("lambda = ", l, " K = ", K, " n1 = ", n1);
        CHECK(at(l, 1, K, n1, 0) == ((n1 % 2) ? -inf : inf));
        CHECK(at(l, 1, K, n1, 1) == ((n1 % 2) ? -inf : inf));
      }
  }
  for (double l : {2.5, 4.0}) {
    for (int K = 1; K <= 6; ++K)
      for (int n1 = 0; n1 <= K; ++n1) {
        INFO("lambda = ", l, " K = ", K, " n1 = ", n1);
        CHECK(at(l, 0, K, n1, 0) == (((K - n1 + 1) % 2) ? -inf : inf));
        CHECK(at(l, 0, K, n1, 1) == (((K - n1) % 2) ? -inf : inf));
      }
  }
}

// ---------------------------------------------------------------- a large finite argument
//
// Above |p1 x| of about 1e3 the generated forms for the parameter-derivatives of the inverse lose
// their value: each is a polynomial in the derivatives of log1prel multiplied by a power of the
// argument, and the power overflows while the derivative underflows, so the product is inf * 0.
// Measured on 2026-11-17: at p1 = 1/2 and x = 1e60 the sixth parameter-derivative came back NaN
// for a value whose size is 7.4e135. The map and its log-Jacobian take a rescaled route there
// instead (scalar_yeo_johnson.hpp), and these are the cases that say so.
//
// The references are the pure parameter-derivatives of order 1 to 6 at the same places, from the
// definition of the inverse differentiated symbolically by SymPy and evaluated at 60 digits by
// mpmath (tools/reference/bijector_yeo_johnson_large.py), so nothing of the C++ went into them.
// The entries that also carry a derivative in the argument are smaller by a factor
// (1 + p1 x)^(-1) per order and underflow to zero at these arguments, which leaves nothing to
// compare; they are checked only for being a number.
constexpr int yj_large_cases = 4;
using yj_large_view = Kokkos::View<double* [2][yj_part_count], Kokkos::LayoutRight>;

void yeo_johnson_large_kernel(Kokkos::View<const double*> p1, Kokkos::View<const double*> xs,
                              yj_large_view out) {
  Kokkos::parallel_for(
      "bijector_yeo_johnson_large", p1.extent(0), KOKKOS_LAMBDA(int i) {
        const auto b = lb::scalar_bijector::yeo_johnson(p1(i));
        yj_fill_partials(b, 1, xs(i), &out(i, 0, 0));
        yj_fill_partials(b, 3, xs(i), &out(i, 1, 0));
      });
}

TEST_CASE("bijector yeo_johnson: the parameter partials at a large finite argument") {
  // a pair on each branch at 1e60, one at 1e300, and one whose sixth derivative overflows
  const double p1_v[yj_large_cases] = {0.5, 0.5, 1.3, 0.3};
  const double x_v[yj_large_cases] = {1e60, -1e60, 1e300, -1e60};
  // d^K_p1 of the inverse, K = 1 to 6, one row per case
  const double ref_map[yj_large_cases][6] = {
      {-1.3646195839908280e+122, 7.5031312194048345e+124, -4.1552650623562085e+127,
       2.3176448061380032e+130, -1.3018305683658143e+133, 7.3635921977184855e+135},
      {-8.0113484847197285e+41, -5.0044104753074661e+43, -3.1918595397974731e+45,
       -2.0773303418883439e+47, -1.3787431928266879e+49, -9.3269195189995701e+50},
      {-2.9367416123614587e+233, 1.2036053180620503e+236, -4.9513705829709825e+238,
       2.0444698029073191e+241, -8.4730735211042176e+243, 3.5244993650430290e+246},
      {-1.2813475326188393e+37, -6.2548109407359975e+38, -3.1256851274652953e+40,
       -1.5977758825561322e+42, -8.3484688410836860e+43, -4.4557524858383806e+45}};
  // and of its log-Jacobian
  const double ref_lj[yj_large_cases][6] = {
      {-547.84783359633118, 2179.3913343853247, -13036.348006311948, 104066.78405049559,
       -1038939.8405049559, 12450382.086059470},
      {61.804698083444847, 81.665523370519055, 162.63968871634675, 432.52065139173949,
       1438.8383852975678, 5746.2232256895855},
      {-409.07567583590603, 628.30031148173072, -1448.7683739981398, 4455.4326130318505,
       -17129.691077380883, 79036.016477828546},
      {48.230357726887513, 56.192034940242178, 98.719412241739140, 231.61893147950236,
       679.81527301629213, 2395.4293081731425}};

  Kokkos::View<double*> p1_d("p1", yj_large_cases), x_d("x", yj_large_cases);
  auto p1_h = Kokkos::create_mirror_view(p1_d);
  auto x_h = Kokkos::create_mirror_view(x_d);
  for (int c = 0; c < yj_large_cases; ++c) p1_h(c) = p1_v[c], x_h(c) = x_v[c];
  Kokkos::deep_copy(p1_d, p1_h);
  Kokkos::deep_copy(x_d, x_h);
  yj_large_view out_d("large", yj_large_cases);
  yeo_johnson_large_kernel(p1_d, x_d, out_d);
  auto out = Kokkos::create_mirror_view(out_d);
  Kokkos::deep_copy(out, out_d);

  // Not an assertion that the entries are merely finite or large: without the rescaled route they
  // are NaN, so requiring a number of every one of them is what distinguishes the two.
  for (int c = 0; c < yj_large_cases; ++c)
    for (int lj = 0; lj < 2; ++lj)
      for (int K = 1; K <= 6; ++K)
        for (int n1 = 0; n1 <= K; ++n1) {
          INFO("case ", c, ", log-Jacobian ", lj, ", K ", K, ", n1 ", n1);
          CHECK_FALSE(std::isnan(out(c, lj, yj_part_offset(K) + n1)));
        }

  // The value of the inverse is exp(S) with S = log(1 + p1 x)/p1, so one unit in the last place
  // of S moves it by a relative |S|, which is 276 at x = 1e60 and 690 at x = 1e300: the bound
  // allows that amplification, measured at 8.4e-14 at the worst of these places, and cannot be
  // tightened by any route that computes the value in double. The log-Jacobian is S times a
  // factor and carries no such amplification: it agrees to one unit in the last place.
  for (int c = 0; c < yj_large_cases; ++c)
    for (int K = 1; K <= 6; ++K) {
      INFO("the inverse, case ", c, ", K ", K);
      CHECK(out(c, 0, yj_part_offset(K) + K) == doctest::Approx(ref_map[c][K - 1]).epsilon(4e-13));
      INFO("its log-Jacobian, case ", c, ", K ", K);
      CHECK(out(c, 1, yj_part_offset(K) + K) == doctest::Approx(ref_lj[c][K - 1]).epsilon(1e-14));
    }
}

// ------------------------------------- a huge argument while the parameter times it stays small
//
// The other side of the same loss, and the one place the rescaled route above cannot reach:
// where p1 is so small that p1 x stays small even at a huge x, the forms carry a power of the
// argument in front of a bracket in the derivatives of log1prel which cancels as the argument of
// log1prel goes to zero, so the power is infinite, the bracket is zero, and the product is NaN.
// Measured on 2026-11-17: the log-Jacobian's partials begin to fail from |x| = 1e103 at order
// six and from 1e154 at orders four and five, and most of the values they lose are FINITE --
// at x = 1e300 and order six, 0 at n1 = 2 and -3.6e301 at n1 = 3 -- so a guard returning signed
// infinities would have been wrong. Above |x| = 1e50 those entries come from a third route
// (scalar_yeo_johnson.hpp), which groups each term so that one power of the argument carries the
// whole of a bounded sum.
//
// At p1 = 0 every entry has a closed form, and these references are SymPy's own limit of the
// definition as p1 goes to zero (tools/reference/bijector_yeo_johnson_large.py), a third route
// to them independent both of the forms and of the identity the repair uses. The comparison is
// exact wherever the value is zero or an infinity, those being the answers at stake, and
// relative elsewhere.
//
// The map needs nothing in this band: at p1 = 0 the inverse is exp(x) - 1, already an infinity
// for every |x| above about 710, so there is no finite value of its own to lose.
constexpr int yj_band_cases = 3;

void yeo_johnson_band_kernel(Kokkos::View<const double*> xs,
                             Kokkos::View<double* [yj_part_count], Kokkos::LayoutRight> out) {
  Kokkos::parallel_for(
      "bijector_yeo_johnson_band", xs.extent(0), KOKKOS_LAMBDA(int i) {
        const auto b = lb::scalar_bijector::yeo_johnson(0.0);
        yj_fill_partials(b, 3, xs(i), &out(i, 0));
      });
}

TEST_CASE("bijector yeo_johnson: the parameter partials at a huge argument with p1 = 0") {
  const double inf = std::numeric_limits<double>::infinity();
  const double x_v[yj_band_cases] = {1e103, 1e150, 1e300};
  // the orders 1 to 6 one after another, entry n1 of order K at yj_part_offset(K) + n1
  const double ref[yj_band_cases][yj_part_count] = {
      // x = 1e103
      {1.0, -5.0e+205, 0.0, -1.0e+103, inf,
       0.0, -1.0, 2.0e+206, -inf, 0.0,
       0.0, 4.0e+103, -inf, inf, 0.0,
       0.0, 4.0, -1.8e+207, inf, -inf,
       0.0, 0.0, 0.0, -3.6e+104, inf,
       -inf, inf},
      // x = 1e150
      {1.0, -5.0e+299, 0.0, -1.0e+150, inf,
       0.0, -1.0, 2.0e+300, -inf, 0.0,
       0.0, 4.0e+150, -inf, inf, 0.0,
       0.0, 4.0, -1.8e+301, inf, -inf,
       0.0, 0.0, 0.0, -3.6e+151, inf,
       -inf, inf},
      // x = 1e300, where three more of the entries have grown past the largest double
      {1.0, -inf, 0.0, -1.0e+300, inf,
       0.0, -1.0, inf, -inf, 0.0,
       0.0, 4.0e+300, -inf, inf, 0.0,
       0.0, 4.0, -inf, inf, -inf,
       0.0, 0.0, 0.0, -3.6e+301, inf,
       -inf, inf}};

  Kokkos::View<double*> x_d("x", yj_band_cases);
  auto x_h = Kokkos::create_mirror_view(x_d);
  for (int c = 0; c < yj_band_cases; ++c) x_h(c) = x_v[c];
  Kokkos::deep_copy(x_d, x_h);
  Kokkos::View<double* [yj_part_count], Kokkos::LayoutRight> out_d("band", yj_band_cases);
  yeo_johnson_band_kernel(x_d, out_d);
  auto out = Kokkos::create_mirror_view(out_d);
  Kokkos::deep_copy(out, out_d);

  for (int c = 0; c < yj_band_cases; ++c)
    for (int K = 1; K <= 6; ++K)
      for (int n1 = 0; n1 <= K; ++n1) {
        const int q = yj_part_offset(K) + n1;
        const double got = out(c, q), want = ref[c][q];
        INFO("x index ", c, ", K ", K, ", n1 ", n1);
        if (!std::isfinite(want) || want == 0.0) {
          // an infinity or an exact zero: compared exactly, since an assertion that merely
          // asked for a number would pass whether the route fires or not
          CHECK(got == want);
        } else {
          CHECK(got == doctest::Approx(want).epsilon(1e-14));
        }
      }
}
