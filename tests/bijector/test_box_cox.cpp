// box_cox: the Box-Cox bijector, g(v) = (v^lambda - 1)/lambda (log v at lambda = 0), against its mpmath
// references (the shared check of param_common.hpp), plus exact special values, the bounds of the domains,
// the image of lambda = 0 and the limits, invalid parameters, inversion, the distances, the derivatives in
// lambda against finite differences, and continuity through lambda = 0.
// Written 2026-11-12: the bounds below are the measured maxima (ulpC, see the
// docs/numerics.md) times about 1.5.
#include "param_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

}  // namespace

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

TEST_CASE("box_cox: components in the absent second parameter are zero") {
  const auto p3 = box(0.7).forward_partials_d3(2.0);
  for (int i = 4; i < 10; ++i) CHECK(p3.c[i] == 0.0);
  const auto q4 = box(0.7).inverse_partials_d4(0.3);
  for (int i = 5; i < 15; ++i) CHECK(q4.c[i] == 0.0);
  const auto r5 = box(0.7).inverse_log_jacobian_partials_d5(0.3);
  for (int i = 6; i < 21; ++i) CHECK(r5.c[i] == 0.0);
}
