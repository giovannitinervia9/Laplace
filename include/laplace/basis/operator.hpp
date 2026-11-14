// Linear differential operators with constant coefficients and the penalties
// they define on a basis (laplace::basis); design decision D8b of 2026-11-14
// (diario.md).
//
// An operator of order m is monic,
//
//   L = D^m + c_{m-1} D^(m-1) + ... + c_0,   P(r) = sum_k c_k r^k (c_m = 1),
//
// with D the derivative in x and P its characteristic polynomial. (Scaling L
// by a constant scales the penalty by its square, which the smoothing
// parameter absorbs.) It is stored by the roots of P with their
// multiplicities: a real root a, or a pair of complex conjugate roots
// a +- i b (b > 0). The coefficients come from multiplying the factors
// (r - a) and (r^2 - 2 a r + a^2 + b^2). The product of two operators is the
// product of the polynomials: the union of the roots, with the multiplicities
// of equal roots added exactly.
//
// The null space of L (the solutions of L f = 0) has dimension m: for each
// root of multiplicity mu and i = 0..mu-1, the functions
//   x'^i e^(a x')            (real root a)
//   x'^i e^(a x') cos(b x'),  x'^i e^(a x') sin(b x')   (pair a +- i b)
// of x' = x - lower (an origin at lower keeps e^(a x') in range).
//
// The penalty int (L f)^2 dx of f = sum_j beta_j b_j is beta' S beta with
//   S = sum_{k,l} c_k c_l G_kl,   G_kl = int b^(k) b^(l)' dx
// (the cross Gram matrices of the basis, exact): operator_penalty.
//
// Everything here runs on the host.
#pragma once

#include <laplace/basis/core.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace laplace::basis {

/**
 * \ingroup basis_operator
 * \brief A root of the characteristic polynomial with its multiplicity.
 *
 * Represents a root \f$r = a + \mathrm{i} b\f$ of the characteristic polynomial. For a real
 * root, \f$b = 0\f$. For complex roots, \f$b > 0\f$ represents the conjugate pair
 * \f$a \pm \mathrm{i} b\f$, which contributes \f$2 \mu\f$ to the total operator order, where
 * \f$\mu\f$ is the multiplicity.
 *
 * \see differential_operator, differential_operator_from_roots()
 */
// a root of the characteristic polynomial: re + i im; im > 0 stands for the
// pair re +- i im, which counts twice in the order
struct differential_operator_root {
  double re = 0.0, im = 0.0;
  int multiplicity = 1;
};

/**
 * \ingroup basis_operator
 * \brief Linear differential operator with constant coefficients.
 *
 * Represents a monic linear differential operator
 * \f[
 * L = D^m + c_{m-1} D^{m-1} + \dots + c_0
 * \f]
 * defined by the roots of its characteristic polynomial \f$P(r) = \sum_{k=0}^m c_k r^k\f$ with \f$c_m = 1\f$.
 * Stored internally by its roots with multiplicities.
 *
 * \see derivative_operator(), oscillator_operator(), harmonic_operator(), operator_penalty()
 */
class differential_operator {
 public:
  /**
   * \ingroup basis_operator
   * \brief One basis function of the operator null space.
   *
   * Represents \f$x'^p e^{a x'} \cos(b x')\f$ (if `sine == false`) or
   * \f$x'^p e^{a x'} \sin(b x')\f$ (if `sine == true`), where \f$p\f$ is `power`,
   * \f$a\f$ is `rate`, \f$b\f$ is `frequency`, and \f$x' = x - \mathrm{lower}\f$.
   * A frequency of zero indicates no trigonometric factor.
   *
   * \see null_space(), null_space_values()
   */
  // one function of the null space: x'^power e^(rate x') times cos(frequency x')
  // (sine = false) or sin(frequency x') (sine = true); frequency 0 means no
  // trigonometric factor
  struct null_space_function {
    double rate = 0.0, frequency = 0.0;
    int power = 0;
    bool sine = false;
  };

  /**
   * \brief Constructs a differential operator from its characteristic roots.
   *
   * \param roots Vector of characteristic roots with multiplicities.
   * \throw std::invalid_argument If any root is non-finite, if \f$\mathrm{im} < 0\f$,
   *        if any multiplicity is less than 1, or if the resulting order is less than 1.
   * \see differential_operator_from_roots()
   */
  explicit differential_operator(std::vector<differential_operator_root> roots) : roots_() {
    for (const auto& r : roots) {
      if (!std::isfinite(r.re) || !std::isfinite(r.im) || r.im < 0.0)
        throw std::invalid_argument("differential_operator: roots must be finite, with im >= 0 (one per conjugate pair)");
      if (r.multiplicity < 1) throw std::invalid_argument("differential_operator: multiplicities must be >= 1");
      add_root(r);
    }
    if (order() < 1) throw std::invalid_argument("differential_operator: the order must be at least 1");
  }

  /**
   * \brief Roots of the characteristic polynomial.
   *
   * \return Const reference to the vector of distinct roots with multiplicities.
   */
  const std::vector<differential_operator_root>& roots() const { return roots_; }

  /**
   * \brief Differential order of the operator.
   *
   * \return Total order \f$m\f$, summing \f$\mu\f$ for real roots and \f$2\mu\f$ for complex conjugate pairs.
   */
  int order() const {
    int m = 0;
    for (const auto& r : roots_) m += (r.im > 0.0 ? 2 : 1) * r.multiplicity;
    return m;
  }

  /**
   * \brief Computes the monic polynomial coefficients \f$c_0, \dots, c_m\f$.
   *
   * Multiplies the linear factors \f$(r - a)\f$ for real roots and quadratic factors
   * \f$(r^2 - 2 a r + a^2 + b^2)\f$ for complex conjugate pairs \f$a \pm \mathrm{i} b\f$.
   *
   * \return Vector of \f$m + 1\f$ coefficients \f$c_0, \dots, c_m\f$ with \f$c_m = 1\f$.
   */
  // c_0..c_m (c_m = 1)
  std::vector<double> coefficients() const {
    std::vector<double> p{1.0};
    auto multiply = [&p](const std::vector<double>& f) {
      std::vector<double> q(p.size() + f.size() - 1, 0.0);
      for (size_t i = 0; i < p.size(); ++i)
        for (size_t j = 0; j < f.size(); ++j) q[i + j] += p[i] * f[j];
      p = std::move(q);
    };
    for (const auto& r : roots_)
      for (int k = 0; k < r.multiplicity; ++k) {
        if (r.im > 0.0) multiply({r.re * r.re + r.im * r.im, -2.0 * r.re, 1.0});
        else multiply({-r.re, 1.0});
      }
    return p;
  }

  /**
   * \brief Canonical functions spanning the null space of the operator.
   *
   * Generates the \f$m\f$ basis functions of the null space for each root with multiplicity \f$\mu\f$
   * and power \f$i = 0, \dots, \mu - 1\f$.
   *
   * \return Vector of \f$m\f$ null-space function descriptors.
   * \see null_space_values()
   */
  std::vector<null_space_function> null_space() const {
    std::vector<null_space_function> f;
    for (const auto& r : roots_)
      for (int i = 0; i < r.multiplicity; ++i) {
        f.push_back({r.re, r.im, i, false});
        if (r.im > 0.0) f.push_back({r.re, r.im, i, true});
      }
    return f;
  }

  /**
   * \brief Evaluates the null-space functions at given points.
   *
   * Evaluates the \f$m\f$ null-space functions at \f$n\f$ coordinates with origin at \f$\mathrm{lower}\f$
   * (\f$x' = x - \mathrm{lower}\f$).
   *
   * \param lower Reference origin coordinate \f$\mathrm{lower}\f$.
   * \param x Pointer to array of \f$n\f$ evaluation points.
   * \param n Number of evaluation points.
   * \param out Pointer to output buffer of size at least \f$n \times m\f$, written column-major:
   *            `out[i + j * n] = f_j(x_i)` for \f$i = 0, \dots, n-1\f$ and \f$j = 0, \dots, m-1\f$.
   * \see null_space()
   */
  // the null-space functions at the n points x (x' = x - lower), column-major:
  // out[i + j n] = f_j(x_i), j = 0..order() - 1
  void null_space_values(double lower, const double* x, int n, double* out) const {
    const auto fs = null_space();
    for (size_t j = 0; j < fs.size(); ++j) {
      const auto& f = fs[j];
      for (int i = 0; i < n; ++i) {
        const double t = x[i] - lower;
        double v = std::pow(t, f.power);
        if (f.rate != 0.0) v *= std::exp(f.rate * t);
        if (f.frequency > 0.0) v *= f.sine ? std::sin(f.frequency * t) : std::cos(f.frequency * t);
        out[i + j * static_cast<size_t>(n)] = v;
      }
    }
  }

  /**
   * \brief Product of two differential operators.
   *
   * Computes \f$L_a L_b\f$ by taking the union of the roots of \f$a\f$ and \f$b\f$,
   * adding the multiplicities of equal roots.
   *
   * \param a First differential operator.
   * \param b Second differential operator.
   * \return Product differential operator of order \f$m_a + m_b\f$.
   */
  friend differential_operator operator*(const differential_operator& a, const differential_operator& b) {
    std::vector<differential_operator_root> r = a.roots_;
    r.insert(r.end(), b.roots_.begin(), b.roots_.end());
    return differential_operator(std::move(r));
  }

 private:
  void add_root(const differential_operator_root& r) {
    for (auto& q : roots_)
      if (q.re == r.re && q.im == r.im) {
        q.multiplicity += r.multiplicity;
        return;
      }
    roots_.push_back(r);
  }
  std::vector<differential_operator_root> roots_;
};

/**
 * \ingroup basis_operator
 * \brief Pure derivative operator \f$D^m\f$.
 *
 * Constructs \f$L = D^m\f$, having characteristic root \f$0\f$ with multiplicity \f$m\f$.
 * Its null space consists of polynomials of degree strictly less than \f$m\f$.
 *
 * \param m Derivative order (\f$m \ge 1\f$).
 * \return Differential operator \f$D^m\f$.
 * \throw std::invalid_argument If \f$m < 1\f$.
 * \see oscillator_operator(), harmonic_operator()
 */
// D^m: the root 0 with multiplicity m; null space the polynomials of degree < m
inline differential_operator derivative_operator(int m) {
  if (m < 1) throw std::invalid_argument("derivative_operator: m must be >= 1");
  return differential_operator({{0.0, 0.0, m}});
}

/**
 * \ingroup basis_operator
 * \brief Oscillator operator annihilating fundamental and harmonic frequencies.
 *
 * Constructs
 * \f[
 * L = \prod_{h=1}^H \left(D^2 + (h \nu)^2\right)
 * \f]
 * with \f$\nu = \frac{2\pi}{T}\f$, of order \f$2H\f$. Its null space is spanned by
 * \f$\cos(h \nu x')\f$ and \f$\sin(h \nu x')\f$ for \f$h = 1, \dots, H\f$.
 *
 * \param period Fundamental period \f$T > 0\f$.
 * \param harmonics Number of harmonics \f$H \ge 1\f$.
 * \return Differential operator of order \f$2H\f$.
 * \throw std::invalid_argument If `period <= 0.0`, is non-finite, or `harmonics < 1`.
 * \see harmonic_operator(), derivative_operator()
 */
// prod_{h=1..H} (D^2 + (h nu)^2), nu = 2 pi / period: order 2H; null space
// cos(h nu x'), sin(h nu x'), h = 1..H
inline differential_operator oscillator_operator(double period, int harmonics = 1) {
  if (!(period > 0.0) || !std::isfinite(period)) throw std::invalid_argument("oscillator_operator: period must be positive");
  if (harmonics < 1) throw std::invalid_argument("oscillator_operator: harmonics must be >= 1");
  const double nu = 6.283185307179586476925286766559 / period;
  std::vector<differential_operator_root> r;
  for (int h = 1; h <= harmonics; ++h) r.push_back({0.0, h * nu, 1});
  return differential_operator(std::move(r));
}

/**
 * \ingroup basis_operator
 * \brief Harmonic operator annihilating the constant function and harmonic frequencies.
 *
 * Constructs
 * \f[
 * L = D \prod_{h=1}^H \left(D^2 + (h \nu)^2\right)
 * \f]
 * with \f$\nu = \frac{2\pi}{T}\f$, of order \f$2H + 1\f$. Its null space is spanned by
 * the constant function \f$1\f$ and \f$\cos(h \nu x')\f$, \f$\sin(h \nu x')\f$ for \f$h = 1, \dots, H\f$.
 *
 * \param period Fundamental period \f$T > 0\f$.
 * \param harmonics Number of harmonics \f$H \ge 1\f$.
 * \return Differential operator of order \f$2H + 1\f$.
 * \throw std::invalid_argument If `period <= 0.0`, is non-finite, or `harmonics < 1`.
 * \see oscillator_operator(), derivative_operator()
 */
// D prod_{h=1..H} (D^2 + (h nu)^2): order 2H + 1; null space the constant and
// cos(h nu x'), sin(h nu x'), h = 1..H
inline differential_operator harmonic_operator(double period, int harmonics = 1) {
  return derivative_operator(1) * oscillator_operator(period, harmonics);
}

/**
 * \ingroup basis_operator
 * \brief Constructs a differential operator from a vector of characteristic roots.
 *
 * \param roots Vector of characteristic roots with multiplicities.
 * \return Differential operator.
 * \throw std::invalid_argument If any root or multiplicity is invalid, or total order is less than 1.
 * \see differential_operator_from_weights(), differential_operator
 */
inline differential_operator differential_operator_from_roots(std::vector<differential_operator_root> roots) {
  return differential_operator(std::move(roots));
}

namespace detail_operator {

// All m roots of r^m + w_{m-1} r^(m-1) + ... + w_0 by the Aberth-Ehrlich
// iteration (simultaneous Newton steps with a repulsion between the
// approximations), started on a circle of the radius s below and stopped when
// every step is below 4 eps of the radius, or after 500 sweeps.
inline std::vector<std::complex<double>> polynomial_roots(const std::vector<double>& w, double s) {
  using C = std::complex<double>;
  const int m = static_cast<int>(w.size());
  std::vector<C> z(m);
  for (int k = 0; k < m; ++k) z[k] = std::polar(s, 6.283185307179586 * (k + 0.25) / m + 0.4);
  for (int sweep = 0; sweep < 500; ++sweep) {
    double step = 0.0;
    for (int k = 0; k < m; ++k) {
      C p = 1.0, dp = 0.0;  // Horner for the monic polynomial and its derivative
      for (int j = m - 1; j >= 0; --j) {
        dp = dp * z[k] + p;
        p = p * z[k] + w[j];
      }
      if (p == C(0.0)) continue;
      const C ratio = p / dp;
      C rep = 0.0;
      for (int j = 0; j < m; ++j)
        if (j != k) rep += 1.0 / (z[k] - z[j]);
      const C dz = ratio / (1.0 - ratio * rep);
      z[k] -= dz;
      step = std::max(step, std::abs(dz));
    }
    if (step <= 4 * 2.220446049250313e-16 * s) break;
  }
  return z;
}

}  // namespace detail_operator

/**
 * \ingroup basis_operator
 * \brief Constructs a differential operator from polynomial coefficient weights.
 *
 * Finds the roots of the monic polynomial \f$P(r) = r^m + \sum_{j=0}^{m-1} w_j r^j\f$
 * via the Aberth-Ehrlich iteration, started on a circle of radius \f$s = 2 \max_j |w_j|^{1/(m-j)}\f$.
 * Approximations closer than \f$10^{-4} s\f$ are grouped into multiple roots with mean values,
 * and components with absolute value \f$\le 10^{-4} s\f$ are set to zero.
 * Distinct roots closer than \f$10^{-4} s\f$ are merged.
 *
 * \param w Vector of weights \f$w_0, \dots, w_{m-1}\f$ for orders \f$0, \dots, m-1\f$ (\f$m \ge 1\f$).
 * \return Differential operator with grouped roots.
 * \throw std::invalid_argument If `w` is empty, contains non-finite values, if conjugate roots
 *        cannot be paired, or if rebuilt coefficients deviate from `w` by more than
 *        \f$10^{-6} \max(\max_k |w_k|, s^{m-j})\f$.
 * \see differential_operator_from_roots(), differential_operator
 */
// L = D^m + w_{m-1} D^(m-1) + ... + w_0 from its weights w_0..w_{m-1} (as
// fda's vec2Lfd). The roots of the characteristic polynomial are computed
// numerically (Aberth-Ehrlich, above) and then grouped: approximations closer
// than 1e-4 s, where s = 2 max_j |w_j|^(1/(m - j)) is Fujiwara's bound on
// the roots, are one root of
// that multiplicity (a root of multiplicity mu is found only to about
// eps^(1/mu) s, 6e-6 s for mu = 3), their mean its value; a real part or an
// imaginary part below 1e-4 s is set to zero. Distinct roots closer than
// 1e-4 s are therefore merged: give such an operator by its roots. Throws if
// the coefficients rebuilt from the roots differ from w by more than 1e-6 of
// the largest |w_j| (or 1e-6 s^(m-j), whichever is larger).
inline differential_operator differential_operator_from_weights(const std::vector<double>& w) {
  const int m = static_cast<int>(w.size());
  if (m < 1) throw std::invalid_argument("differential_operator_from_weights: at least one weight");
  for (double v : w)
    if (!std::isfinite(v)) throw std::invalid_argument("differential_operator_from_weights: weights must be finite");
  double s = 0.0;
  for (int j = 0; j < m; ++j) s = std::max(s, std::pow(std::fabs(w[j]), 1.0 / (m - j)));
  if (s == 0.0) return derivative_operator(m);  // all weights zero: D^m
  s *= 2.0;  // Fujiwara's bound: every root lies within 2 max_j |w_j|^(1/(m - j))
  const double tol = 1e-4 * s;
  auto z = detail_operator::polynomial_roots(w, s);
  // group the approximations
  std::vector<int> group(m, -1);
  std::vector<differential_operator_root> roots;
  std::vector<std::complex<double>> centre;
  for (int k = 0; k < m; ++k) {
    if (group[k] >= 0) continue;
    std::complex<double> sum = z[k];
    int count = 1;
    group[k] = static_cast<int>(centre.size());
    for (int j = k + 1; j < m; ++j)
      if (group[j] < 0 && std::abs(z[j] - z[k]) <= tol) {
        group[j] = group[k];
        sum += z[j];
        ++count;
      }
    std::complex<double> c = sum / static_cast<double>(count);
    double re = std::fabs(c.real()) <= tol ? 0.0 : c.real();
    double im = std::fabs(c.imag()) <= tol ? 0.0 : c.imag();
    centre.push_back(c);
    if (im < 0.0) continue;  // the conjugate with im > 0 stands for the pair
    roots.push_back({re, im, count});
  }
  differential_operator op(std::move(roots));
  if (op.order() != m)
    throw std::invalid_argument("differential_operator_from_weights: the roots could not be paired (order " +
                                std::to_string(op.order()) + " instead of " + std::to_string(m) + ")");
  const auto c = op.coefficients();
  double wmax = 0.0;
  for (double v : w) wmax = std::max(wmax, std::fabs(v));
  for (int j = 0; j < m; ++j)
    if (std::fabs(c[j] - w[j]) > 1e-6 * std::max(wmax, std::pow(s, m - j)))
      throw std::invalid_argument("differential_operator_from_weights: the roots do not reproduce weight " +
                                  std::to_string(j) + "; give the operator by its roots");
  return op;
}

namespace detail_operator {
// the degree of the polynomials of a polynomial basis (B-spline, Legendre),
// above which its derivatives vanish; -1 for the others
template <class B>
int polynomial_degree(const B& b) {
  if constexpr (requires(const B& x) { x.degree; }) return b.degree;
  else if constexpr (requires(const B& x) { x.max_degree; }) return b.max_degree;
  else return -1;
}
}  // namespace detail_operator

/**
 * \ingroup basis_operator
 * \brief Computes the penalty matrix of a differential operator on a univariate basis.
 *
 * Evaluates the \f$K \times K\f$ penalty matrix
 * \f[
 * S = \sum_{k=0}^m \sum_{l=0}^m c_k c_l G_{kl}
 * \f]
 * where \f$c_k\f$ are the operator coefficients and \f$G_{kl}\f$ are the cross Gram matrices
 * of derivative orders \f$k\f$ and \f$l\f$ computed by `basis.cross_gram(k, l, ...)`.
 * The operator order \f$m\f$ must be at most 6, and for polynomial bases, at most the basis degree.
 *
 * \tparam B Univariate basis type providing `dimension()` and `cross_gram(k, l, double*)`.
 * \param basis Univariate basis instance.
 * \param op Differential operator \f$L\f$.
 * \param out Pointer to output array for the \f$K \times K\f$ dense matrix in column-major order.
 * \throw std::invalid_argument If operator order \f$m > 6\f$, or if \f$m\f$ exceeds the polynomial
 *        degree of a polynomial basis.
 * \see differential_operator
 */
// S = sum_{k,l} c_k c_l G_kl, dense K x K column-major. The order must be at
// most 6 (the cross Gram matrices go to order 6) and, for a polynomial basis,
// at most its degree (above it D^m b = 0 and the penalty would not be the one
// of L).
template <class B>
void operator_penalty(const B& basis, const differential_operator& op, double* out) {
  const int m = op.order(), K = basis.dimension();
  if (m > 6) throw std::invalid_argument("operator_penalty: the order of the operator must be at most 6");
  const int deg = detail_operator::polynomial_degree(basis);
  if (deg >= 0 && m > deg)
    throw std::invalid_argument("operator_penalty: the order of the operator (" + std::to_string(m) +
                                ") exceeds the degree of the basis (" + std::to_string(deg) + ")");
  const auto c = op.coefficients();
  const size_t KK = static_cast<size_t>(K) * K;
  std::fill(out, out + KK, 0.0);
  std::vector<double> g(KK);
  for (int k = 0; k <= m; ++k)
    for (int l = 0; l <= m; ++l) {
      const double w = c[k] * c[l];
      if (w == 0.0) continue;
      basis.cross_gram(k, l, g.data());
      for (size_t i = 0; i < KK; ++i) out[i] += w * g[i];
    }
}

}  // namespace laplace::basis
