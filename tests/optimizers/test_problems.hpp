#pragma once

// The battery of test problems of the optimizers (note/optimizers.md,
// decision 10). Each problem is a curvature_objective with an analytic value,
// gradient and Hessian, a start x0, a known minimizer x_star and its value
// f_star, and a constant `offset` added to f (f_star includes it): with
// offsets 1, 1e3, 1e6 a run meets the rounding floor of the objective, which
// a battery with f* = 0 never shows. The derivatives are checked against
// central differences in test_optimizers.cpp.
//
// Moré, Garbow and Hillstrom (1981), ACM TOMS 7, 17-41: extended Rosenbrock
// (21), Powell singular (13), Beale (5), Brown badly scaled (4), helical
// valley (7), Wood (14), with their standard starts. Besides: ill-conditioned
// quadratics, a saddle with an indefinite Hessian at the start, the noisy
// objective sum(p^2) + 100 sin(1e20 p_1) and an exploded start (cosh at 700).

#include <laplace/optimizers/objective.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace optimizer_test_problems {

namespace lo = laplace::optimizers;
using point = lo::optimizer_point;
using vector = lo::optimizer_vector;
using matrix = lo::dense_curvature_matrix;

// The three members of the objective from one evaluate(x, g, H) of the
// derived problem, which writes g and H when the pointers are not null and
// returns f without the offset.
template <class P>
struct problem_base {
  using curvature_type = matrix;
  std::string name;
  std::vector<double> x0, x_star;
  double f_star = 0.0;  // includes the offset
  double offset = 0.0;

  int size() const { return static_cast<int>(x0.size()); }
  double value(const point& x) { return offset + self().evaluate(x, nullptr, nullptr); }
  double value_gradient(const point& x, const vector& g) { return offset + self().evaluate(x, &g, nullptr); }
  double value_gradient_curvature(const point& x, const vector& g, matrix& H) {
    return offset + self().evaluate(x, &g, &H);
  }

 private:
  P& self() { return static_cast<P&>(*this); }
};

// f = sum_i r_i(x)^2 from residuals: g = 2 sum r_i grad r_i,
// H = 2 sum (grad r_i grad r_i^T + r_i hess r_i). The derived problem gives
// residual(x, i, grad, hess) for i < residual_count, writing grad (n) and
// hess (n x n, column-major) when not null.
template <class P>
struct least_squares_problem : problem_base<P> {
  int residual_count = 0;
  double evaluate(const point& x, const vector* g, matrix* H) const {
    const int n = static_cast<int>(x.extent(0));
    std::vector<double> grad(n), hess(n * n);
    if (g)
      for (int i = 0; i < n; ++i) (*g)(i) = 0.0;
    if (H)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) (*H)(i, j) = 0.0;
    double f = 0.0;
    for (int k = 0; k < residual_count; ++k) {
      std::fill(grad.begin(), grad.end(), 0.0);
      std::fill(hess.begin(), hess.end(), 0.0);
      const double r = static_cast<const P&>(*this).residual(x, k, g ? grad.data() : nullptr,
                                                            H ? hess.data() : nullptr);
      f += r * r;
      if (g)
        for (int i = 0; i < n; ++i) (*g)(i) += 2.0 * r * grad[i];
      if (H)
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < n; ++i) (*H)(i, j) += 2.0 * (grad[i] * grad[j] + r * hess[i + j * n]);
    }
    return f;
  }
};

// sum over pairs (a, b) = (x_2i, x_2i+1) of 100 (b - a^2)^2 + (1 - a)^2; n even;
// start (-1.2, 1, -1.2, 1, ...), minimizer (1, ..., 1). n = 2 is Rosenbrock.
struct extended_rosenbrock_problem : problem_base<extended_rosenbrock_problem> {
  explicit extended_rosenbrock_problem(int n, double added = 0.0) {
    name = "extended Rosenbrock n = " + std::to_string(n);
    for (int i = 0; i < n; ++i) x0.push_back(i % 2 == 0 ? -1.2 : 1.0);
    x_star.assign(n, 1.0);
    offset = added;
    f_star = added;
  }
  double evaluate(const point& x, const vector* g, matrix* H) const {
    const int n = static_cast<int>(x.extent(0));
    if (H)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) (*H)(i, j) = 0.0;
    double f = 0.0;
    for (int i = 0; i + 1 < n; i += 2) {
      const double a = x(i), b = x(i + 1), t = b - a * a;
      f += 100.0 * t * t + (1.0 - a) * (1.0 - a);
      if (g) {
        (*g)(i) = -400.0 * a * t - 2.0 * (1.0 - a);
        (*g)(i + 1) = 200.0 * t;
      }
      if (H) {
        (*H)(i, i) = 1200.0 * a * a - 400.0 * b + 2.0;
        (*H)(i, i + 1) = (*H)(i + 1, i) = -400.0 * a;
        (*H)(i + 1, i + 1) = 200.0;
      }
    }
    return f;
  }
};

// r = (x1 + 10 x2, sqrt5 (x3 - x4), (x2 - 2 x3)^2, sqrt10 (x1 - x4)^2); start
// (3, -1, 0, 1), minimizer 0 with a singular Hessian there.
struct powell_singular_problem : least_squares_problem<powell_singular_problem> {
  explicit powell_singular_problem(double added = 0.0) {
    name = "Powell singular";
    residual_count = 4;
    x0 = {3.0, -1.0, 0.0, 1.0};
    x_star = {0.0, 0.0, 0.0, 0.0};
    offset = added;
    f_star = added;
  }
  double residual(const point& x, int k, double* grad, double* hess) const {
    const double s5 = std::sqrt(5.0), s10 = std::sqrt(10.0);
    switch (k) {
      case 0:
        if (grad) grad[0] = 1.0, grad[1] = 10.0;
        return x(0) + 10.0 * x(1);
      case 1:
        if (grad) grad[2] = s5, grad[3] = -s5;
        return s5 * (x(2) - x(3));
      case 2: {
        const double t = x(1) - 2.0 * x(2);
        if (grad) grad[1] = 2.0 * t, grad[2] = -4.0 * t;
        if (hess) hess[1 + 4 * 1] = 2.0, hess[1 + 4 * 2] = hess[2 + 4 * 1] = -4.0, hess[2 + 4 * 2] = 8.0;
        return t * t;
      }
      default: {
        const double t = x(0) - x(3);
        if (grad) grad[0] = 2.0 * s10 * t, grad[3] = -2.0 * s10 * t;
        if (hess) hess[0] = 2.0 * s10, hess[0 + 4 * 3] = hess[3] = -2.0 * s10, hess[3 + 4 * 3] = 2.0 * s10;
        return s10 * t * t;
      }
    }
  }
};

// r_i = y_i - x1 (1 - x2^i), y = (1.5, 2.25, 2.625); start (1, 1), minimizer (3, 0.5).
struct beale_problem : least_squares_problem<beale_problem> {
  explicit beale_problem(double added = 0.0) {
    name = "Beale";
    residual_count = 3;
    x0 = {1.0, 1.0};
    x_star = {3.0, 0.5};
    offset = added;
    f_star = added;
  }
  double residual(const point& x, int k, double* grad, double* hess) const {
    const double y[3] = {1.5, 2.25, 2.625};
    const int i = k + 1;
    const double p = std::pow(x(1), i), p1 = i * std::pow(x(1), i - 1);
    if (grad) grad[0] = -(1.0 - p), grad[1] = x(0) * p1;
    if (hess) {
      hess[0 + 2 * 1] = hess[1] = p1;
      hess[1 + 2 * 1] = i > 1 ? x(0) * i * (i - 1) * std::pow(x(1), i - 2) : 0.0;
    }
    return y[k] - x(0) * (1.0 - p);
  }
};

// r = (x1 - 1e6, x2 - 2e-6, x1 x2 - 2); start (1, 1), minimizer (1e6, 2e-6).
struct brown_badly_scaled_problem : least_squares_problem<brown_badly_scaled_problem> {
  explicit brown_badly_scaled_problem(double added = 0.0) {
    name = "Brown badly scaled";
    residual_count = 3;
    x0 = {1.0, 1.0};
    x_star = {1e6, 2e-6};
    offset = added;
    f_star = added;
  }
  double residual(const point& x, int k, double* grad, double* hess) const {
    switch (k) {
      case 0:
        if (grad) grad[0] = 1.0;
        return x(0) - 1e6;
      case 1:
        if (grad) grad[1] = 1.0;
        return x(1) - 2e-6;
      default:
        if (grad) grad[0] = x(1), grad[1] = x(0);
        if (hess) hess[0 + 2 * 1] = hess[1] = 1.0;
        return x(0) * x(1) - 2.0;
    }
  }
};

// r = (10 (x3 - 10 theta), 10 (rho - 1), x3) with rho = |(x1, x2)| and
// 2 pi theta = atan(x2 / x1) (+ pi when x1 < 0); start (-1, 0, 0), minimizer (1, 0, 0).
struct helical_valley_problem : least_squares_problem<helical_valley_problem> {
  explicit helical_valley_problem(double added = 0.0) {
    name = "helical valley";
    residual_count = 3;
    x0 = {-1.0, 0.0, 0.0};
    x_star = {1.0, 0.0, 0.0};
    offset = added;
    f_star = added;
  }
  double residual(const point& x, int k, double* grad, double* hess) const {
    const double x1 = x(0), x2 = x(1), x3 = x(2), twopi = 2.0 * std::numbers::pi;
    const double r2 = x1 * x1 + x2 * x2, rho = std::sqrt(r2);
    switch (k) {
      case 0: {
        const double theta = x1 > 0.0   ? std::atan(x2 / x1) / twopi
                             : x1 < 0.0 ? std::atan(x2 / x1) / twopi + 0.5
                                        : (x2 >= 0.0 ? 0.25 : -0.25);
        if (grad) grad[0] = -100.0 * (-x2 / (twopi * r2)), grad[1] = -100.0 * (x1 / (twopi * r2)), grad[2] = 10.0;
        if (hess) {
          const double r4 = twopi * r2 * r2;
          hess[0] = -100.0 * (2.0 * x1 * x2 / r4);
          hess[0 + 3 * 1] = hess[1] = -100.0 * ((x2 * x2 - x1 * x1) / r4);
          hess[1 + 3 * 1] = -100.0 * (-2.0 * x1 * x2 / r4);
        }
        return 10.0 * (x3 - 10.0 * theta);
      }
      case 1: {
        if (grad) grad[0] = 10.0 * x1 / rho, grad[1] = 10.0 * x2 / rho;
        if (hess) {
          const double r3 = rho * r2;
          hess[0] = 10.0 * x2 * x2 / r3;
          hess[0 + 3 * 1] = hess[1] = -10.0 * x1 * x2 / r3;
          hess[1 + 3 * 1] = 10.0 * x1 * x1 / r3;
        }
        return 10.0 * (rho - 1.0);
      }
      default:
        if (grad) grad[2] = 1.0;
        return x3;
    }
  }
};

// r = (10 (x2 - x1^2), 1 - x1, sqrt90 (x4 - x3^2), 1 - x3, sqrt10 (x2 + x4 - 2),
// (x2 - x4) / sqrt10); start (-3, -1, -3, -1), minimizer (1, 1, 1, 1).
struct wood_problem : least_squares_problem<wood_problem> {
  explicit wood_problem(double added = 0.0) {
    name = "Wood";
    residual_count = 6;
    x0 = {-3.0, -1.0, -3.0, -1.0};
    x_star = {1.0, 1.0, 1.0, 1.0};
    offset = added;
    f_star = added;
  }
  double residual(const point& x, int k, double* grad, double* hess) const {
    const double s90 = std::sqrt(90.0), s10 = std::sqrt(10.0);
    switch (k) {
      case 0:
        if (grad) grad[0] = -20.0 * x(0), grad[1] = 10.0;
        if (hess) hess[0] = -20.0;
        return 10.0 * (x(1) - x(0) * x(0));
      case 1:
        if (grad) grad[0] = -1.0;
        return 1.0 - x(0);
      case 2:
        if (grad) grad[2] = -2.0 * s90 * x(2), grad[3] = s90;
        if (hess) hess[2 + 4 * 2] = -2.0 * s90;
        return s90 * (x(3) - x(2) * x(2));
      case 3:
        if (grad) grad[2] = -1.0;
        return 1.0 - x(2);
      case 4:
        if (grad) grad[1] = s10, grad[3] = s10;
        return s10 * (x(1) + x(3) - 2.0);
      default:
        if (grad) grad[1] = 1.0 / s10, grad[3] = -1.0 / s10;
        return (x(1) - x(3)) / s10;
    }
  }
};

// A deterministic uniform on [0, 1) (64-bit LCG of Knuth's MMIX, top 53 bits),
// the same on every platform.
struct test_uniform {
  std::uint64_t state;
  double operator()() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * 0x1.0p-53;
  }
};

// f = (1/2) (x - x*)^T A (x - x*) with A = Q diag(lambda) Q^T, the lambda
// log-spaced from 1 to kappa and Q the Householder reflection I - 2 v v^T / v^T v
// of a pseudo-random v; x* = (1, 2, ..., n), start 0.
struct ill_conditioned_quadratic_problem : problem_base<ill_conditioned_quadratic_problem> {
  std::vector<double> a;  // n x n, column-major
  std::vector<double> eigenvalues;
  ill_conditioned_quadratic_problem(int n, double kappa, double added = 0.0) {
    name = "quadratic n = " + std::to_string(n) + ", kappa = " + std::to_string(kappa);
    x0.assign(n, 0.0);
    for (int i = 0; i < n; ++i) x_star.push_back(i + 1.0);
    offset = added;
    f_star = added;
    test_uniform u{12345u + static_cast<std::uint64_t>(n)};
    std::vector<double> v(n);
    double vv = 0.0;
    for (double& vi : v) vi = u() - 0.5, vv += vi * vi;
    eigenvalues.resize(n);
    for (int i = 0; i < n; ++i) eigenvalues[i] = n > 1 ? std::pow(kappa, double(i) / (n - 1)) : 1.0;
    a.assign(n * n, 0.0);
    // A = Q L Q with Q symmetric: A_ij = sum_k Q_ik l_k Q_kj
    auto q = [&](int i, int k) { return (i == k ? 1.0 : 0.0) - 2.0 * v[i] * v[k] / vv; };
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = 0; k < n; ++k) s += q(i, k) * eigenvalues[k] * q(k, j);
        a[i + j * n] = s;
      }
  }
  double evaluate(const point& x, const vector* g, matrix* H) const {
    const int n = static_cast<int>(x.extent(0));
    double f = 0.0;
    for (int i = 0; i < n; ++i) {
      double ai = 0.0;  // (A (x - x*))_i
      for (int j = 0; j < n; ++j) ai += a[i + j * n] * (x(j) - x_star[j]);
      f += 0.5 * (x(i) - x_star[i]) * ai;
      if (g) (*g)(i) = ai;
    }
    if (H)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) (*H)(i, j) = a[i + j * n];
    return f;
  }
};

// f = x1^2 + x2^4 / 4 - x2^2 / 2: a saddle at 0 and minima (0, +-1) with
// f = -1/4. The start (1, 0) has H = diag(2, -1), indefinite, and g_2 = 0:
// Newton goes to the saddle, a trust region must leave along x2.
struct saddle_problem : problem_base<saddle_problem> {
  explicit saddle_problem(double added = 0.0) {
    name = "saddle";
    x0 = {1.0, 0.0};
    x_star = {0.0, 1.0};  // or (0, -1)
    offset = added;
    f_star = added - 0.25;
  }
  double evaluate(const point& x, const vector* g, matrix* H) const {
    const double a = x(0), b = x(1);
    if (g) (*g)(0) = 2.0 * a, (*g)(1) = b * b * b - b;
    if (H) (*H)(0, 0) = 2.0, (*H)(0, 1) = (*H)(1, 0) = 0.0, (*H)(1, 1) = 3.0 * b * b - 1.0;
    return a * a + 0.25 * b * b * b * b - 0.5 * b * b;
  }
};

// f = sum p_i^2 + 100 sin(1e20 p_1): the sine is noise of amplitude 100 at the
// rounding scale of p_1, so no method can reach a stationary point; a run must
// end not converged. Start (1, ..., 1); x_star = 0 and f_star = NaN (no
// minimum is defined). The derivatives are exact but useless.
struct noisy_problem : problem_base<noisy_problem> {
  explicit noisy_problem(int n) {
    name = "noisy";
    x0.assign(n, 1.0);
    x_star.assign(n, 0.0);
    f_star = std::numeric_limits<double>::quiet_NaN();
  }
  double evaluate(const point& x, const vector* g, matrix* H) const {
    const int n = static_cast<int>(x.extent(0));
    double f = 100.0 * std::sin(1e20 * x(0));
    for (int i = 0; i < n; ++i) f += x(i) * x(i);
    if (g) {
      for (int i = 0; i < n; ++i) (*g)(i) = 2.0 * x(i);
      (*g)(0) += 100.0 * 1e20 * std::cos(1e20 * x(0));
    }
    if (H) {
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) (*H)(i, j) = i == j ? 2.0 : 0.0;
      (*H)(0, 0) -= 100.0 * 1e40 * std::sin(1e20 * x(0));
    }
    return f;
  }
};

// f = sum cosh(x_i), minimizer 0 with f* = n; started at 700, where f is
// 5e303 per coordinate (finite) and every Newton step moves by about 1: an
// exploded start that a run with a modest budget must end not converged.
struct exploded_cosh_problem : problem_base<exploded_cosh_problem> {
  explicit exploded_cosh_problem(int n, double added = 0.0) {
    name = "exploded cosh";
    x0.assign(n, 700.0);
    x_star.assign(n, 0.0);
    offset = added;
    f_star = added + n;
  }
  double evaluate(const point& x, const vector* g, matrix* H) const {
    const int n = static_cast<int>(x.extent(0));
    double f = 0.0;
    for (int i = 0; i < n; ++i) {
      f += std::cosh(x(i));
      if (g) (*g)(i) = std::sinh(x(i));
    }
    if (H)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) (*H)(i, j) = i == j ? std::cosh(x(i)) : 0.0;
    return f;
  }
};

}  // namespace optimizer_test_problems
