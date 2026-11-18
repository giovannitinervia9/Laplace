// powerexp_distrib (laplace::distrib): the univariate power exponential distribution (also called
// the generalized normal of the first kind, or the exponential power law) in its location mu, its
// standard deviation sigma > 0 and its shape p > 0. With
//
//   c(p) = sqrt(Gamma(1/p) / Gamma(3/p)),   s = sigma c(p),   z = (y - mu) / s,
//   f(y) = p / (2 s Gamma(1/p)) exp(-|z|^p),
//
// sigma carries the whole variance (Var Y = s^2 Gamma(3/p)/Gamma(1/p) = sigma^2 for every p) and p
// only the tails, the convention of pseudohuber_distrib. The law is the Gaussian of standard
// deviation sigma at p = 2, where c = sqrt(2), and the Laplace of scale sigma / sqrt(2) at p = 1,
// where c = 1 / sqrt(2); as p grows c(p) tends to sqrt(3), NOT to one, because Gamma(1/p) ~ p and
// Gamma(3/p) ~ p/3, so the limit law is the uniform on [mu - sigma sqrt(3), mu + sigma sqrt(3)],
// whose variance is sigma^2. Below p = 1 the density has a cusp at the location and is not
// log-concave; the law is defined there all the same, and nothing here refuses it.
//
// The log-density is written as one function of zeta = (y - mu)/sigma and p plus two functions of
// one parameter each, which is the form the derivatives are assembled in:
//
//   l = Lambda(zeta, p) + g(p) - log sigma,
//   Lambda(zeta, p) = -|zeta|^p exp(A(p)),
//   A(p) = -p log c(p) = -(p/2) (lgamma(1/p) - lgamma(3/p)),
//   g(p) = log p - log 2 - (3/2) lgamma(1/p) + (1/2) lgamma(3/p),
//
// both evaluated in the equivalent forms of shape_derivs, through lgamma(1 + 1/p) and
// lgamma(1 + 3/p), which never form the logarithms that cancel between them.
//
// g and -log sigma add their own derivatives to the pure-p and the pure-sigma component, as the
// other location-scale families with one shape parameter do; Lambda is differentiated in the three
// parameters at once, because it FACTORIZES over them:
//
//   Lambda = -|y - mu|^p sigma^-p exp(A(p)),
//
// so a derivatives in mu hit |y - mu|^p, b derivatives in sigma hit sigma^-p, and the derivatives
// in p spread over the three factors by Leibniz's rule. This is why the table of (zeta, p)
// derivatives that a location-scale family usually carries to (mu, sigma) through
// detail_location_scale::mixed_shape_dk is not used here: that route multiplies each derivative in
// zeta by a polynomial in zeta, and at the location, where a derivative in zeta is infinite, a
// factor zeta would turn it into a NaN. The factorized form has no such product.
//
// REGULARITY IN y, the one delicate point. Since
//
//   d^j/dz^j |z|^p = p (p-1) ... (p-j+1) |z|^(p-j) sgn(z)^j,
//
// the derivative of order j at z = 0 is finite and zero while j < p and infinite once j > p, so the
// regularity of l in y (and in mu) depends on p, which is a runtime value. y_regularity is a static
// descriptor and is therefore declared 0, the worst case over p; the TRUE order is returned at
// runtime in the continuous_order field of y_kinks() and of param_kinks(0, y), as ceil(p) - 1,
// which is floor(p) away from the integers (where the floor(p)-th derivative is continuous and
// zero, the exponent p - floor(p) being positive) and p - 1 at an integer p, the order at which an
// odd integer p jumps (at p = 3 the third derivative is 6 sgn(z)). The one exception is p == 2.0
// exactly, the Gaussian, where l is a polynomial in zeta and the order is infinitely_smooth; no
// other even integer is tested for, an exact equality on a runtime double being fragile while p
// moves under an optimizer (Giovanni, 2026-11-18). Being conservative at p = 4, 6, ... costs panel
// granularity in the quadratures and nothing else.
//
// One consumer reads the static field: independent_product_penalty.hpp sets
// beta_regularity = D::y_regularity and asks for penalty_kink_geometry::per_coordinate unless the
// law is infinitely_smooth, so a power exponential used as a penalty always takes the kink
// geometry, at p >= 2 as well. A kink treated as present is never wrong, only slower, and this is
// the price of a static descriptor.
//
// THE DERIVATIVES. Write fall(p, a) = p (p-1) ... (p-a+1) and rise(p, b) = p (p+1) ... (p+b-1) for
// the falling and the rising factorial, r = y - mu, L = log|zeta| and
// V(p) = (p - a) log|r| - (p + b) log sigma + A(p). Then
//
//   d^a/dmu^a |r|^p = (-1)^a sgn(r)^a fall(p, a) |r|^(p-a),
//   d^b/dsigma^b sigma^-p = (-1)^b rise(p, b) sigma^(-p-b),
//
// and V' = L + A'(p) while V^(m) = A^(m)(p) for m >= 2, so exp(V) is differentiated by the complete
// Bell polynomials Y_m of those. One Leibniz rule over the three factors gives every component:
//
//   d^a/dmu^a d^b/dsigma^b d^c/dp^c Lambda
//     = -(-1)^(a+b) sgn(r)^a |zeta|^(p-a) exp(A(p)) sigma^(-a-b)
//       sum_(i+j+k=c) c!/(i! j! k!) [d^i fall(p, a)/dp^i] [d^j rise(p, b)/dp^j] Y_k,
//
// where |zeta|^(p-a) exp(A(p)) sigma^(-a-b) is exp(V), computed as exp(p (L - log c(p)) - a L)
// sigma^(-a-b). The magnitude depends on a alone, which is what makes the limits at the location
// simple: it is |zeta|^(p-a), so only the derivatives in mu can be singular there, and the number
// of derivatives in sigma never changes whether they are.
//
// A sym_tensor<3, K> holds the components of total order exactly K, so the assembly walks the pairs
// (a, b) with a + b <= K and takes c = K - a - b. Walking a + b + c <= K instead writes each slot
// several times, because pos3(a, b, c) with a + b + c < K is a position in the smaller tensor and
// lands on another component of this one.
//
// The derivatives of A and of g go through lgamma(1 + 1/p) and lgamma(1 + 3/p), whose p-derivatives
// are one Faa di Bruno composition each through v = 1 + a/p with the polygamma functions of v
// (polygamma.hpp, orders 0 to 5 here). The derivation is checked against SymPy in
// tools/reference/powerexp_forms.py.
//
// Numerics. |z|^p is never formed as exp(p log|zeta| + A): for large p both terms are huge and
// nearly opposite, so it is formed as exp(p (L - log c(p))), where the difference is of order one,
// and exp(V) as exp(p (L - log c) - a L) sigma^(-a-b). Each order a carries its own exponential
// rather than |zeta|^-a times a common one, because the common one underflows near the location while
// |zeta|^(p-a) with a > p is the huge number that is wanted there. Where the sum over (i, j, k) is
// exactly zero (p an integer below a, so that fall(p, a) = 0 and |zeta|^p is a polynomial of degree
// below a on each side) the component is zero and the exponential is not multiplied in, which would
// be 0 * infinity. At
// |z|^p above about 709 the value overflows and l is -infinity, the density having underflowed.
//
// At zeta = 0 exactly the components are taken from the limits rather than from the formula, where
// L = -infinity: a component with no derivative in mu is zero, Lambda vanishing identically in
// (sigma, p) at r = 0; an odd number of derivatives in mu gives zero by the convention sgn(0) = 0
// of laplace1.hpp (the mean of the two one-sided derivatives, whose magnitudes are equal); and an
// even number a is zero while a < p (an exponential beats the powers of log|zeta|),
// -(-1)^b fall(p, a) rise(p, b) exp(A) sigma^(-a-b) at a = p with no derivative in p, and infinite
// otherwise, with the sign of the term carrying the highest power of L, which is the one with the
// first non-zero derivative of fall. Where p is an integer below a, fall(p, a) is zero and so is
// the component, correctly: |zeta|^p is then a polynomial of degree below a. The infinite case is a
// fact about the family and not about the point: d^2 l / dzeta^2 at zeta = 0 is 0 above p = 2,
// -2 exp(A) at p = 2 and infinite below it, so its derivative in p is infinite at p = 2, where l
// itself is a polynomial in zeta.
//
// THE CUMULATIVE FUNCTION is closed through the regularized incomplete gamma (incomplete_gamma.hpp):
// with t = |z|^p, which carries 1/p of the shape of a Gamma(1/p, 1) variable,
//
//   F(y) = 1/2 + sgn(z) P(1/p, t) / 2,
//
// so the far tail is half the upper regularized function and no side is formed as a difference. The
// quantile inverts the same relation (gamma_p_inv, gamma_q_inv). The derivative TENSORS of log F
// and log(1 - F) in the parameters are not supplied: they would need the derivatives of P(1/p, t)
// in its shape mixed with those in t, and incomplete_gamma_deriv.hpp carries the shape derivatives
// to order five, one short of the six the three parameters need. The generic quadrature route
// (generic_cdf.hpp) answers for them.
//
// THE EXPECTED INFORMATION is closed. With a = 1/p and q = |z|^p, which is Gamma(a, 1) under the
// law, the scores in the natural parametrization (mu, s, p) are (p/s)|z|^(p-1) sgn(z), (p q - 1)/s
// and a^2 psi(1 + a) - a q log q; the moments of q and of q log q give
//
//   I_mu mu   = p^2 Gamma(2 - 1/p) / (s^2 Gamma(1/p)),
//   I_s s     = p / s^2,   I_s p = -(1 + psi(1 + a)) / (p s),
//   I_p p     = -a^4 psi(1 + a)^2 + a^3 (1 + a) (psi'(2 + a) + psi(2 + a)^2),
//
// with I_mu s = I_mu p = 0 by symmetry, and the Jacobian of s = sigma c(p) carries them to
// (mu, sigma, p) through kappa = (log c)'(p) = (3 psi(3/p) - psi(1/p)) / (2 p^2):
//
//   I_mu mu unchanged,   I_sigma sigma = p / sigma^2,
//   I_sigma p = (p kappa - (1 + psi(1 + a)) / p) / sigma,
//   I_p p = I^nat_p p - 2 kappa (1 + psi(1 + a)) / p + p kappa^2.
//
// I_mu mu is the integral of |z|^(2p-2) against the density and is finite only for p > 1/2; at
// p <= 1/2 the component is +infinity, which the law reports rather than a finite number. The
// (sigma, p) block stays finite for every p > 0. I_sigma p is exactly zero at p = 2: the standard
// deviation and the shape are orthogonal at the Gaussian. The closed form is checked against the
// quadrature of the score products in tools/reference/powerexp_forms.py. Its derivatives in the
// parameters are not supplied; the generic route (generic_info.hpp) answers for them.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/incomplete_inverse.hpp>
#include <laplace/math/polygamma.hpp>
#include <laplace/math/special.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_powerexp {

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double sign_of_power(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// position in sym_tensor<3, a + b + c> of the sorted multi-index with a zeros, b ones and c twos
LAPLACE_HD_FORCEINLINE constexpr int pos3(int a, int b, int c) {
  const int k = a + b + c;
  int pos = 0, prev = 0;
  for (int j = 0; j < k; ++j) {
    const int v_end = j < a ? 0 : (j < a + b ? 1 : 2);
    for (int v = prev; v < v_end; ++v) pos += binomial_coef(3 - v + k - j - 2, k - j - 1);
    prev = v_end;
  }
  return pos;
}

// d^n/dp^n f(g(p)) for n = 1..6, with f[1..n] and g[1..n] the derivatives of the two functions
LAPLACE_HD_FORCEINLINE double chain(int n, const double* f, const double* g) {
  switch (n) {
    case 1: return math::chain_rule_d1(f, g);
    case 2: return math::chain_rule_d2(f, g);
    case 3: return math::chain_rule_d3(f, g);
    case 4: return math::chain_rule_d4(f, g);
    case 5: return math::chain_rule_d5(f, g);
    default: return math::chain_rule_d6(f, g);
  }
}

// G[m] = d^m/dp^m lgamma(1 + a/p), m = 0..K: one Faa di Bruno composition through v = 1 + a/p,
// whose derivatives are d^k v / dp^k = a (-1)^k k! / p^(k+1), with the polygamma functions of v
// outside. The argument is 1 + a/p and not a/p on purpose: lgamma(a/p) = lgamma(1 + a/p) - log(a/p),
// and the term -log(a/p) is exactly the part that cancels between lgamma(1/p) and lgamma(3/p) in
// log c(p), and between lgamma(1/p) and log p in g(p) (see shape_derivs). Leaving it out here
// means it is never formed.
template <int K>
LAPLACE_HD void lgamma1p_over_p_derivs(double a, double p, double* G) {
  const double v = 1.0 + a / p;
  G[0] = math::lgamma(v);
  if constexpr (K >= 1) {
    double f[K + 1], g[K + 1];
    f[0] = 0.0;
    g[0] = 0.0;
    double pk = p, kf = 1.0;
    for (int k = 1; k <= K; ++k) {
      f[k] = math::polygamma(k - 1, v);
      kf *= k;
      pk *= p;
      g[k] = (k % 2 == 0 ? a : -a) * kf / pk;
    }
    for (int m = 1; m <= K; ++m) G[m] = chain(m, f, g);
  }
}

// lambda(p) = log c(p) and the derivatives of A(p) = -p lambda(p) and of g(p), orders 0..K, through
// G1 and G3, the derivatives of lgamma(1 + 1/p) and lgamma(1 + 3/p). With
// lgamma(v) = lgamma(1 + v) - log v the two functions of the header comment become
//
//   lambda(p) = (1/2) (lgamma(1 + 1/p) - lgamma(1 + 3/p) + log 3),
//   g(p)      = -log 2 - lgamma(1 + 1/p) - lambda(p),
//
// exact identities in which the logarithms that cancel (log(1/p) - log(3/p) = -log 3 in lambda, and
// log p against the -log(1/p) inside lgamma(1/p) in g) have been cancelled by hand. Measured at the
// worst point of the reference run (order 5 at p = 4), carrying them through lgamma(1/p) and
// lgamma(3/p) instead costs a factor 9 in the pure-p component of the derivative: there the pole
// terms of lgamma(0.25) and lgamma(0.75) are several times larger than their difference.
// A = -p lambda is one Leibniz rule against the factor p (two terms, p being linear).
template <int K>
LAPLACE_HD void shape_derivs(double p, double* A, double* B, double& logc) {
  double G1[K + 1], G3[K + 1], lam[K + 1];
  lgamma1p_over_p_derivs<K>(1.0, p, G1);
  lgamma1p_over_p_derivs<K>(3.0, p, G3);
  lam[0] = 0.5 * (G1[0] - G3[0] + 1.0986122886681097821);  // + log 3
  for (int m = 1; m <= K; ++m) lam[m] = 0.5 * (G1[m] - G3[m]);
  logc = lam[0];
  A[0] = -p * lam[0];
  B[0] = -0.69314718055994530942 - G1[0] - lam[0];
  for (int m = 1; m <= K; ++m) {
    A[m] = -(p * lam[m] + m * lam[m - 1]);
    B[m] = -G1[m] - lam[m];
  }
}

// t = |z|^p = exp(p (log|zeta| - log c(p))), the argument of the regularized incomplete gamma and
// the magnitude of Lambda; both terms of the exponent are of order one, where p log|zeta| + A(p)
// would be two huge nearly opposite numbers at large p.
LAPLACE_HD_FORCEINLINE double abs_z_power(double az, double p, double logc) {
  return az == 0.0 ? 0.0 : math::exp(p * (math::log(az) - logc));
}

// T[n * (K + 1) + i] = d^i/dp^i of a product of n consecutive factors starting at p, i, n = 0..K:
// with step = -1 the falling factorial fall(p, n) = p (p-1) ... (p-n+1), with step = +1 the rising
// one rise(p, n) = p (p+1) ... (p+n-1). The coefficients in p follow from the recurrence
// prod(p, n + 1) = (p + step n) prod(p, n), and they are integers, so nothing is lost in building
// them; the i-th derivative is then the exact derivative of that polynomial.
template <int K>
LAPLACE_HD void factorial_table(double p, int step, double* T) {
  constexpr int W = K + 2;
  double co[(K + 1) * W] = {};
  co[0] = 1.0;
  for (int n = 0; n < K; ++n) {
    const double* prev = co + n * W;
    double* cur = co + (n + 1) * W;
    for (int m = 0; m <= n; ++m) {
      cur[m + 1] += prev[m];
      cur[m] += step * n * prev[m];
    }
  }
  for (int n = 0; n <= K; ++n) {
    const double* c = co + n * W;
    for (int i = 0; i <= K; ++i) {
      double s = 0.0, pw = 1.0, ff = fact(i);  // pw = p^(m-i), ff = m!/(m-i)!
      for (int m = i; m <= n; ++m) {
        s += c[m] * ff * pw;
        pw *= p;
        ff = ff * (m + 1) / (m + 1 - i);
      }
      T[n * (K + 1) + i] = s;
    }
  }
}

// the multinomial coefficient c! / (i! j! k!), i + j + k = c <= 6, exact in a double
LAPLACE_HD_FORCEINLINE double multinomial(int c, int i, int j, int k) {
  return fact(c) / (fact(i) * fact(j) * fact(k));
}

}  // namespace detail_powerexp

/**
 * \ingroup distrib_powerexp
 * \brief Univariate power exponential distribution in location \f$\mu\f$, standard deviation \f$\sigma > 0\f$ and shape \f$p > 0\f$.
 *
 * Implements the density \f$f(y) = \frac{p}{2 s \Gamma(1/p)} \exp(-|z|^p)\f$ with
 * \f$z = (y - \mu)/s\f$ and \f$s = \sigma c(p)\f$, \f$c(p) = \sqrt{\Gamma(1/p)/\Gamma(3/p)}\f$, so
 * that \f$\operatorname{Var} Y = \sigma^2\f$ for every \f$p\f$. The Gaussian of standard deviation
 * \f$\sigma\f$ at \f$p = 2\f$ and the Laplace of scale \f$\sigma/\sqrt{2}\f$ at \f$p = 1\f$ are
 * exact special cases.
 * \see distrib_powerexp, distrib_core
 */
struct powerexp_distrib {
  /// Location parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Standard deviation \f$\sigma \in (0, \infty)\f$.
  double sigma;
  /// Shape (tail exponent) \f$p \in (0, \infty)\f$.
  double p;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (3).
  static constexpr int n_params = 3;
  /**
   * \ingroup distrib_powerexp
   * \brief Differentiability order of the log-density in \f$y\f$: 0, the worst case over \f$p\f$.
   *
   * The true order depends on the runtime value of \f$p\f$ and is reported by y_kinks().
   */
  static constexpr int y_regularity = 0;
  /// Descriptors of \c mu, \c sigma and \c p.
  static constexpr distrib_param_info params[3] = {
      {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"p", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates \f$\log c(p) = \tfrac{1}{2}(\log\Gamma(1/p) - \log\Gamma(3/p))\f$, the logarithm of the ratio of the natural scale to \f$\sigma\f$.
   * \return \f$\log c(p)\f$.
   */
  LAPLACE_HD double log_scale_ratio() const {
    // (1/2)(lgamma(1/p) - lgamma(3/p)) written through lgamma(1 + v) = lgamma(v) + log v, so that the
    // two logarithms of v, which cancel to -log 3, are never formed (shape_derivs)
    return 0.5 * (math::lgamma(1.0 + 1.0 / p) - math::lgamma(1.0 + 3.0 / p) + 1.0986122886681097821);
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates the natural scale \f$s = \sigma c(p)\f$ of the kernel \f$\exp(-|(y-\mu)/s|^p)\f$.
   * \return \f$s\f$.
   */
  LAPLACE_HD double natural_scale() const { return sigma * math::exp(log_scale_ratio()); }

  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma, p)\f$.
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density value, \f$-\infty\f$ where \f$|z|^p\f$ overflows.
   */
  LAPLACE_HD double logpdf(double y) const {
    double A[1], B[1], logc;
    detail_powerexp::shape_derivs<0>(p, A, B, logc);
    const double az = Kokkos::fabs(y - mu) / sigma;
    return -detail_powerexp::abs_z_power(az, p, logc) + B[0] - math::log(sigma);
  }

  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<3, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dk(double y) const {
    using detail_powerexp::fact;
    using detail_powerexp::multinomial;
    using detail_powerexp::pos3;
    using detail_powerexp::sign_of_power;
    double A[K + 1], B[K + 1], logc;
    detail_powerexp::shape_derivs<K>(p, A, B, logc);
    // F = the derivatives of fall(p, a) = p (p-1) ... (p-a+1), from the a derivatives in mu;
    // G = those of rise(p, b) = p (p+1) ... (p+b-1), from the b derivatives in sigma
    double F[(K + 1) * (K + 1)], G[(K + 1) * (K + 1)];
    detail_powerexp::factorial_table<K>(p, -1, F);
    detail_powerexp::factorial_table<K>(p, +1, G);
    const double r = y - mu, az = Kokkos::fabs(r) / sigma;
    sym_tensor<3, K> t;
    if (az == 0.0) {
      // The limits at the location (header comment): every component with no derivative in mu is
      // zero, since Lambda vanishes identically in (sigma, p) there; an odd number of them is zero
      // by sgn(0) = 0; and an even number a is zero while a < p, finite at a = p with no derivative
      // in p, and infinite otherwise, with the sign the limit fixes.
      const double eA = math::exp(A[0]);
      for (int a = 0; a <= K; ++a) {
        // the first non-zero derivative of fall carries the largest power of log|zeta|
        int imin = -1;
        for (int i = 0; i <= K; ++i)
          if (F[a * (K + 1) + i] != 0.0) {
            imin = i;
            break;
          }
        const bool singular = a >= 2 && a % 2 == 0 && p <= double(a) && imin >= 0;
        const double Fi = imin >= 0 ? F[a * (K + 1) + imin] : 0.0;
        for (int b = 0; a + b <= K; ++b) {
          const int c = K - a - b;
          double v = 0.0;
          if (singular && imin <= c) {
            const int m = c - imin;
            const double sb = -sign_of_power(b);  // -(-1)^(a + b) with a even
            if (p == double(a) && m == 0) {
              double isk = 1.0;
              for (int k = 0; k < a + b; ++k) isk /= sigma;
              v = sb * Fi * G[b * (K + 1)] * eA * isk;
            } else {
              v = sb * (Fi > 0.0 ? 1.0 : -1.0) * sign_of_power(m) * distrib_infinity;
            }
          }
          t[pos3(a, b, c)] = v;
        }
      }
    } else {
      const double L = math::log(az);
      const double pw = p * (L - logc);  // log |z|^p, both terms of order one
      double w[K + 1], one[K + 1], Y[K + 1];
      w[0] = 0.0;
      for (int m = 1; m <= K; ++m) w[m] = (m == 1) ? L + A[1] : A[m];
      for (int m = 0; m <= K; ++m) one[m] = 1.0;
      Y[0] = 1.0;
      for (int m = 1; m <= K; ++m) Y[m] = detail_powerexp::chain(m, one, w);  // complete Bell
      const double sgr = r > 0.0 ? 1.0 : -1.0;
      for (int a = 0; a <= K; ++a) {
        const double sa = (a % 2 == 0) ? 1.0 : sgr;
        const double za = math::exp(pw - a * L);  // |zeta|^(p-a)
        for (int b = 0; a + b <= K; ++b) {
          const int c = K - a - b;
          double isk = 1.0;
          for (int k = 0; k < a + b; ++k) isk /= sigma;
          const double pre = -sign_of_power(a + b) * sa * za * isk;
          double s = 0.0;
          for (int i = 0; i <= c; ++i) {
            const double Fi = F[a * (K + 1) + i];
            if (Fi == 0.0) continue;  // p an integer below a: no exponential is multiplied in
            for (int j = 0; i + j <= c; ++j)
              s += multinomial(c, i, j, c - i - j) * Fi * G[b * (K + 1) + j] * Y[c - i - j];
          }
          t[pos3(a, b, c)] = (s == 0.0) ? 0.0 : pre * s;
        }
      }
    }
    t[pos3(0, 0, K)] += B[K];
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    t[pos3(0, K, 0)] += sign_of_power(K) * fact(K - 1) * isk;
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \sigma, p)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
  LAPLACE_HD sym_tensor<3, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
  LAPLACE_HD sym_tensor<3, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
  LAPLACE_HD sym_tensor<3, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
  LAPLACE_HD sym_tensor<3, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
  LAPLACE_HD sym_tensor<3, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma, p)\f$.
  LAPLACE_HD sym_tensor<3, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates mixed partial derivatives with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in \f$(\mu, \sigma, p)\f$.
   *
   * \f$\log f\f$ reaches \f$y\f$ and \f$\mu\f$ only through \f$y - \mu\f$, so
   * \f$\partial/\partial y = -\partial/\partial\mu\f$ and the result is \f$(-1)^J\f$ times the
   * component of logpdf_dk() of order \f$J + K\f$ with \f$J\f$ further indices in \f$\mu\f$.
   * \tparam J Derivative order in \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order in the parameters (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<3, K> of derivatives.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<3, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<3, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<3, K> r;
      const double s = detail_powerexp::sign_of_power(J);
      for (int b = 0; b <= K; ++b)
        for (int c = 0; b + c <= K; ++c)
          r[detail_powerexp::pos3(K - b - c, b, c)] =
              s * t[detail_powerexp::pos3(K - b - c + J, b, c)];
      return r;
    }
  }

  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates the expected Fisher information \f$I(\mu, \sigma, p)\f$ in closed form.
   *
   * \f$I_{\mu\sigma} = I_{\mu p} = 0\f$ by symmetry, and \f$I_{\mu\mu}\f$ is \f$+\infty\f$ for
   * \f$p \le 1/2\f$, where \f$\mathbb{E}[|z|^{2p-2}]\f$ diverges. The formulas are in the header
   * comment.
   * \return Symmetric tensor \c sym_tensor<3, 2>, the pairs in lexicographic order.
   */
  LAPLACE_HD sym_tensor<3, 2> expected_info() const {
    // Every gamma function is taken at 1 + v rather than at v, with lgamma(1 + v) = lgamma(v) + log v
    // and psi(1 + v) = psi(v) + 1/v: in kappa the poles 3/(3a) and 1/a cancel exactly, so
    // 3 psi(3a) - psi(a) = 3 psi(1 + 3a) - psi(1 + a); and p^2 Gamma(2 - a)/Gamma(a) is
    // p Gamma(2 - a)/Gamma(1 + a), since Gamma(a) = Gamma(1 + a)/a.
    const double a = 1.0 / p;
    const double g1 = math::lgamma(1.0 + a);
    const double s = sigma * math::exp(log_scale_ratio());
    const double psi1a = math::digamma(1.0 + a), psi2a = math::digamma(2.0 + a);
    const double kappa = (3.0 * math::digamma(1.0 + 3.0 * a) - psi1a) / (2.0 * p * p);
    const double nat_pp =
        -a * a * a * a * psi1a * psi1a +
        a * a * a * (1.0 + a) * (math::trigamma(2.0 + a) + psi2a * psi2a);
    sym_tensor<3, 2> t;
    t[0] = p <= 0.5 ? distrib_infinity : p * math::exp(math::lgamma(2.0 - a) - g1) / (s * s);
    t[3] = p / (sigma * sigma);
    t[4] = (p * kappa - (1.0 + psi1a) * a) / sigma;
    t[5] = nat_pp - 2.0 * kappa * (1.0 + psi1a) * a + p * kappa * kappa;
    return t;
  }

  // F(y) = 1/2 + sgn(z) P(1/p, t)/2 with t = |z|^p; the far side is half the upper regularized
  // function, so no tail is a difference of numbers near 1/2.
  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates the cumulative distribution function \f$F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    const double t = detail_powerexp::abs_z_power(Kokkos::fabs(y - mu) / sigma, p, log_scale_ratio());
    return y < mu ? 0.5 * math::gamma_q(1.0 / p, t) : 0.5 + 0.5 * math::gamma_p(1.0 / p, t);
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double t = detail_powerexp::abs_z_power(Kokkos::fabs(y - mu) / sigma, p, log_scale_ratio());
    return y > mu ? 0.5 * math::gamma_q(1.0 / p, t) : 0.5 + 0.5 * math::gamma_p(1.0 / p, t);
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates \f$\log F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double t = detail_powerexp::abs_z_power(Kokkos::fabs(y - mu) / sigma, p, log_scale_ratio());
    if (y < mu) return -0.69314718055994530942 + math::log_gamma_q(1.0 / p, t);
    return -0.69314718055994530942 + Kokkos::log1p(math::gamma_p(1.0 / p, t));
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double t = detail_powerexp::abs_z_power(Kokkos::fabs(y - mu) / sigma, p, log_scale_ratio());
    if (y > mu) return -0.69314718055994530942 + math::log_gamma_q(1.0 / p, t);
    return -0.69314718055994530942 + Kokkos::log1p(math::gamma_p(1.0 / p, t));
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Evaluates the quantile function \f$F^{-1}(u)\f$.
   *
   * Inverts \f$F\f$ through the inverse regularized incomplete gamma: \f$Q(1/p, t) = 2u\f$ below
   * the median and \f$P(1/p, t) = 2u - 1\f$ above it.
   * \param u Probability \f$u \in (0, 1)\f$.
   * \return Quantile \f$y\f$.
   */
  LAPLACE_HD double quantile(double u) const {
    const double a = 1.0 / p, s = natural_scale();
    if (u < 0.5) return mu - s * Kokkos::pow(math::gamma_q_inv(a, 2.0 * u), a);
    return mu + s * Kokkos::pow(math::gamma_p_inv(a, 2.0 * u - 1.0), a);
  }

  /// \ingroup distrib_powerexp
  /// \brief Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// \ingroup distrib_powerexp
  /// \brief Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// \ingroup distrib_powerexp
  /// \brief Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_powerexp
  /// \brief Typical spread (\f$\sigma\f$).
  LAPLACE_HD double scale() const { return sigma; }

  /**
   * \ingroup distrib_powerexp
   * \brief Non-smooth point of \f$\theta_j\f$ placed by \f$y\f$: the location at \f$\mu = y\f$.
   *
   * For \f$j = 0\f$ (\c mu) one kink at \f$\mu = y\f$ with the runtime order of kink_order();
   * \c sigma and \c p have none.
   * \param j Parameter index.
   * \param y Observation value.
   * \return Kinks descriptor \c distrib_kinks.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const {
    distrib_kinks k;
    if (j == 0) {
      k.count = 1;
      k.point[0] = y;
      k.continuous_order = kink_order();
    }
    return k;
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Non-smooth point of \f$\log f\f$ in \f$y\f$: the location \f$y = \mu\f$, with the order that \f$p\f$ gives.
   * \return Kinks descriptor \c distrib_kinks with \c continuous_order from kink_order().
   */
  LAPLACE_HD distrib_kinks y_kinks() const {
    distrib_kinks k;
    k.count = 1;
    k.point[0] = mu;
    k.continuous_order = kink_order();
    return k;
  }
  /**
   * \ingroup distrib_powerexp
   * \brief Highest derivative order of \f$\log f\f$ that stays continuous at the location.
   *
   * \f$\lceil p \rceil - 1\f$, which is \f$\lfloor p \rfloor\f$ away from the integers and
   * \f$p - 1\f$ at an integer \f$p\f$, and \c infinitely_smooth at \f$p = 2\f$ exactly, where
   * \f$\log f\f$ is a polynomial in \f$y\f$. No other even integer is tested for; the order is
   * then one less than the truth, which costs panel granularity only. Capped at \f$10^6\f$.
   * \return The order, or \c infinitely_smooth.
   */
  LAPLACE_HD int kink_order() const {
    if (p == 2.0) return infinitely_smooth;
    return int(Kokkos::fmin(Kokkos::ceil(p) - 1.0, 1.0e6));
  }

  /**
   * \ingroup distrib_powerexp
   * \brief Starting value from a sample summary: the mean, the standard deviation and \f$p = 2\f$, the Gaussian.
   * \param s Sample summary computed by summarize_sample().
   * \return A \c powerexp_distrib with parameters strictly inside their open domains.
   * \see distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  powerexp_distrib start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.mean, params[0]), start_inside(Kokkos::sqrt(s.variance), params[1]),
            start_inside(2.0, params[2])};
  }
};

}  // namespace laplace::distrib
