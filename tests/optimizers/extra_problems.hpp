#pragma once

// Two problems beside the battery of test_problems.hpp, for the quasi-Newton
// methods and multistart: a non-smooth one (a smooth method must not
// declare convergence on it) and a multimodal one with minima of different
// values.

#include "test_problems.hpp"

#include <cmath>

namespace optimizer_test_problems {

// f = sum |x_i| with the subgradient sign(x_i) (0 at 0); minimizer 0, start
// (1, -2, 3, ...): not differentiable at the minimizer.
struct abs_sum_problem {
  std::vector<double> x0, x_star;
  double f_star = 0.0;
  explicit abs_sum_problem(int n) {
    for (int i = 0; i < n; ++i) x0.push_back((i % 2 == 0 ? 1.0 : -1.0) * (i + 1));
    x_star.assign(n, 0.0);
  }
  double value(const point& x) {
    double f = 0.0;
    for (std::size_t i = 0; i < x.extent(0); ++i) f += std::abs(x(i));
    return f;
  }
  double value_gradient(const point& x, const vector& g) {
    for (std::size_t i = 0; i < x.extent(0); ++i) g(i) = x(i) > 0.0 ? 1.0 : x(i) < 0.0 ? -1.0 : 0.0;
    return value(x);
  }
};

// Styblinski-Tang, f = (1/2) sum (x_i^4 - 16 x_i^2 + 5 x_i): each coordinate
// has a global minimizer a (about -2.9035) and a local one b (about 2.7468),
// the roots of 4 t^3 - 32 t + 5 = 0, so 2^n minima with n + 1 distinct
// values (set by the number of coordinates at b). Global minimum at
// (a, ..., a). The start (2.5, ..., 2.5) lies in the basin of (b, ..., b), the
// worst minimum.
struct styblinski_tang_problem {
  std::vector<double> x0;
  double a = 0.0, b = 0.0;  // the global and the local coordinate minimizers
  explicit styblinski_tang_problem(int n) : x0(n, 2.5) {
    auto root = [](double t) {  // Newton on 4 t^3 - 32 t + 5
      for (int k = 0; k < 50; ++k) t -= (4.0 * t * t * t - 32.0 * t + 5.0) / (12.0 * t * t - 32.0);
      return t;
    };
    a = root(-3.0), b = root(3.0);
  }
  static double term(double t) { return 0.5 * (t * t * t * t - 16.0 * t * t + 5.0 * t); }
  // the value of a minimum with `local` coordinates at b and the rest at a
  double minimum_value(int local) const {
    const int n = static_cast<int>(x0.size());
    return (n - local) * term(a) + local * term(b);
  }
  double value(const point& x) {
    double f = 0.0;
    for (std::size_t i = 0; i < x.extent(0); ++i) f += term(x(i));
    return f;
  }
  double value_gradient(const point& x, const vector& g) {
    for (std::size_t i = 0; i < x.extent(0); ++i) g(i) = 0.5 * (4.0 * x(i) * x(i) * x(i) - 32.0 * x(i) + 5.0);
    return value(x);
  }
};

}  // namespace optimizer_test_problems
